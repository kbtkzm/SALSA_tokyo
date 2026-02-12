# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import os
import numpy as np
import math
from scipy.linalg import circulant
from logging import getLogger

logger = getLogger()

class Generator(ABC):
    def __init__(self, params):
        pass

    @abstractmethod
    def generate(self, rng):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass


class ModularMultiply(Generator):
    def __init__(self, params, secret):
        super().__init__(params)
        self.Q = params.Q
        self.S = secret
        self.N = params.N
        assert len(self.S) == self.N

    def generate(self, rng):
        a = rng.randint(0, self.Q, self.N)
        result = [np.dot(a, self.S) % self.Q]
        return a, result

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

#### RLWE DATA ####

class RLWE(Generator):
    def __init__(self, params, rng):
        super().__init__(params)
        self.N = params.N
        self.Q = params.Q
        self.rng = rng
        self.sparsity= params.sparsity
        self.density = params.density
        self.hamming = params.hamming #if not self.hamming_curriculum else 1
        self.error = params.error
        self.sigma = params.sigma
        self.maxQ_prob = params.maxQ_prob
        self.percQ_bound = params.percQ_bound
        self.correctQ = params.correctQ
        self.matrix_mode = getattr(params, "matrix_mode", "circulant")
        self.a_reduced_mode = getattr(params, "a_reduced_mode", "elements")
        self.q2_correction = np.vectorize(self.q2_correct)
        self.fixed_secret_seed = params.fixed_secret_seed
        if self.fixed_secret_seed >= 0:
            self.secret_rng = np.random.default_rng(int(self.fixed_secret_seed))
        else:
            self.secret_rng = self.rng
        self.ab_reduced_A, self.ab_reduced_b, self.ab_reduced_path = self._load_ab_reduced(
            params.ab_reduced_source, params.a_reduced_source, params.b_reduced_source
        )
        self.a_reduced_pool, self.a_reduced_path = self._load_a_reduced_pool(
            params.a_reduced_source if self.ab_reduced_A is None else ""
        )
        self._logged_a_reduced_usage = False

        # if density is greater than 0, set hamming weight by it. 
        if self.density > 0: 
            ham = round(self.N * self.density) 
            self.hamming = ham

        # curriculum parameters
        self.secrets = self.getSecrets(params)
        logger.info(f'secrets: {self.secrets}')

        # reuse data? 
        self.reuse = params.reuse
        if self.reuse:
            self.reuse_samples = np.zeros(shape=(params.num_reuse_samples,self.N,self.N+1)) # N+1 allows space to store B
            self.reuse_counter = np.zeros(shape=params.num_reuse_samples) - 1
            self.times_reused = params.times_reused
            self.K = params.K
        else:
            self.reuse_samples, self.times_reused, self.reuse_counter = None, None, None

    def _load_a_reduced_pool(self, source_path):
        if not source_path:
            return None, None
        path = os.path.expanduser(source_path)
        if os.path.isdir(path):
            path = os.path.join(path, "A_reduced.npy")
        path = os.path.abspath(path)
        try:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"A_reduced.npy not found at: {path}")
            if path.endswith(".npz"):
                data = np.load(path)
                if "A_reduced" not in data:
                    raise ValueError(f"Missing 'A_reduced' in npz file: {path}")
                arr = data["A_reduced"]
            else:
                arr = np.load(path)
            pool = np.array(arr, dtype=np.int64)
            if self.matrix_mode == "general" and self.a_reduced_mode == "matrix":
                if pool.ndim != 3 or pool.shape[1:] != (self.N, self.N):
                    raise ValueError(
                        f"A_reduced.npy must have shape (num,{self.N},{self.N}) for matrix mode; got {pool.shape}"
                    )
            else:
                pool = pool.reshape(-1)
            if pool.size == 0:
                raise ValueError(f"A_reduced.npy is empty: {path}")
        except Exception as exc:
            logger.warning(
                f"Failed to load A_reduced from {path}: {exc}. Falling back to uniform a."
            )
            return None, None
        logger.info(f"Loaded A_reduced pool from {path} (size={pool.size})")
        return pool, path

    def _load_ab_reduced(self, ab_source, a_source, b_source):
        if not ab_source and not b_source:
            return None, None, None
        if ab_source:
            base = os.path.abspath(os.path.expanduser(ab_source))
            a_path = os.path.join(base, "A_reduced.npy")
            b_path = os.path.join(base, "b_reduced.npy")
        else:
            a_path = os.path.abspath(os.path.expanduser(a_source)) if a_source else ""
            b_path = os.path.abspath(os.path.expanduser(b_source)) if b_source else ""
            if os.path.isdir(a_path):
                a_path = os.path.join(a_path, "A_reduced.npy")
            if os.path.isdir(b_path):
                b_path = os.path.join(b_path, "b_reduced.npy")
        if not a_path or not b_path:
            return None, None, None
        if not (os.path.isfile(a_path) and os.path.isfile(b_path)):
            raise FileNotFoundError(f"A_reduced.npy or b_reduced.npy not found at: {a_path}, {b_path}")
        A = np.load(a_path)
        B = np.load(b_path)
        if A.ndim != 3 or A.shape[1:] != (self.N, self.N):
            raise ValueError(f"A_reduced.npy shape {A.shape} does not match (num,{self.N},{self.N})")
        if B.ndim != 2 or B.shape[1] != self.N:
            raise ValueError(f"b_reduced.npy shape {B.shape} does not match (num,{self.N})")
        if A.shape[0] != B.shape[0]:
            raise ValueError(f"A_reduced.npy and b_reduced.npy have different counts: {A.shape[0]} vs {B.shape[0]}")
        logger.info(f"Loaded A_reduced/b_reduced samples from {os.path.dirname(a_path)} (count={A.shape[0]})")
        return A.astype(np.int64), B.astype(np.int64), os.path.dirname(a_path)

    def getSecrets(self, params):
        if getattr(params, "secret_source", ""):
            path = os.path.abspath(os.path.expanduser(params.secret_source))
            if os.path.isdir(path):
                path = os.path.join(path, "secret.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"secret.npy not found at: {path}")
            s = np.load(path)
            s = np.array(s, dtype=np.int64).reshape(-1)
            if s.shape[0] != self.N:
                raise ValueError(f"secret.npy shape {s.shape} does not match N={self.N}")
            secrets = [s]
            return secrets
        secrets = [self.genSecretKey(params.secrettype, self.N)]
        return secrets

    def genSecretKey(self, secret, N):
        rng = self.secret_rng
        if secret == "b":
            # sample secret uniformly from {0, 1}
            if self.hamming == 0:
                s = np.vectorize(lambda x: 1 if x <= self.sparsity else 0)(rng.uniform(size=N))
                while self.N > 1 and np.sum(s) < 2: # make sure you have at least 2 nonzero elements.
                    s[rng.integers(N)] = 1
            else:
                s = np.zeros(shape=N, dtype=np.int64)
                for _ in range(self.hamming):
                    setit = False
                    while not setit:
                        idx = rng.integers(N, size=1)
                        if s[idx] != 1:
                            s[idx] = 1
                            setit = True
        elif secret == "g":
            s = rng.normal(0, self.sigma, size=N).round()
        elif secret == "u":
            s = rng.integers(0, self.Q-1, endpoint=True, size=N)
        elif secret == "t":
            # sample secret uniformly from {-1, 0, 1}
            s = rng.integers(-1, 1, endpoint=True, size=N)
        return s

    def generate(self, rng, idx, currN=-1):
        if self.reuse:
            if self.K > 1:
                return self.combine_reused_samples(rng, idx, currN)
            else:
                return self.get_reused_sample(rng, idx, currN)
        else:
            return self.get_sample(rng, idx, currN)
        
    def combine_reused_samples(self, rng, idx, currN):
        '''
        Combines the reused samples depending on the K level.
        '''
        A_s = np.zeros(shape=(self.K, self.N, self.N), dtype=np.int64)
        B_s = np.zeros(shape=(self.K, self.N), dtype=np.int64)
        for i in range(self.K):
            a,b = self.get_reused_sample(rng, idx, currN)
            A_s[i,:,:] = a
            B_s[i,:] = b
        k_s = rng.choice([-1,0,1], self.K, replace=True).reshape((-1,) + (1,)*(2)).astype(np.int64)
        while np.all(k_s == 0):
            k_s = rng.choice([-1,0,1], self.K, replace=True).reshape((-1,) + (1,)*(2)).astype(np.int64)
        return np.sum(A_s * k_s, axis=0) % self.Q, np.sum(B_s * np.squeeze(k_s, axis=1), axis=0) % self.Q

    def get_reused_sample(self, rng, idx, currN=-1):
        ''' 
        Code to faciliate sample reuse. 
        '''
        # Choose a random sample
        sample_idx = rng.randint(0, self.reuse_samples.shape[0])
        curr_count = self.reuse_counter[sample_idx]
        # If the reuse counter is -1 or times_reused, generate a new sample and put it in the reuse samples array at this index
        if (curr_count == -1) or (curr_count >= self.times_reused):
            A, B = self.get_sample(rng, idx, currN)
            self.reuse_samples[sample_idx, :, :self.N] = A
            self.reuse_samples[sample_idx, :,self.N:] = np.expand_dims(B,1)
            self.reuse_counter[sample_idx] = 0
        # Return the sample at this index
        self.reuse_counter[sample_idx] += 1 / self.K
        a,b = self.reuse_samples[sample_idx, :, :self.N].astype(np.int64), np.squeeze(self.reuse_samples[sample_idx, :, self.N:]).astype(np.int64)
        return a,b 

    def q2_correct(self, x):
        if x <= -self.Q/2:
            x = x+self.Q
        elif x >= self.Q/2:
            x = x-self.Q
        return x

    def get_sample(self, rng, idx, currN=-1):
        # Use passed-in N if it isn't 0.
        N = currN if currN > 0 else self.N
        if self.ab_reduced_A is not None:
            if self.matrix_mode != "general":
                logger.warning("ab_reduced_source provided but matrix_mode is not general; using general for this run.")
            j = rng.randint(0, self.ab_reduced_A.shape[0])
            return self.ab_reduced_A[j], self.ab_reduced_b[j]
        if self.a_reduced_pool is not None:
            if not self._logged_a_reduced_usage:
                logger.info(
                    f"Sampling a from A_reduced pool (path={self.a_reduced_path}, size={self.a_reduced_pool.size})"
                )
                self._logged_a_reduced_usage = True
            if self.matrix_mode == "general":
                if self.a_reduced_mode == "matrix" and self.a_reduced_pool.ndim == 3:
                    c = self.a_reduced_pool[rng.randint(0, self.a_reduced_pool.shape[0])].astype(np.int64) % self.Q
                else:
                    # elements mode: sample entries from pool to build A
                    pool = self.a_reduced_pool.reshape(-1)
                    idxs = rng.integers(0, pool.size, size=(N, N))
                    c = pool[idxs].astype(np.int64) % self.Q
                assert (np.min(c) >= 0) and (np.max(c) < self.Q)
                if self.error:
                    e = np.int64(rng.normal(0, self.sigma, size=self.N).round())
                    b = (c @ self.secrets[idx] + e) % self.Q
                else:
                    b = (c @ self.secrets[idx]) % self.Q
                if self.correctQ:
                    b = self.q2_correction(b)
                return c, b
            else:
                a = rng.choice(self.a_reduced_pool, size=N, replace=True).astype(np.int64)
                a = a % self.Q
                c = circulant(a)
                tri = np.triu_indices(N, 1)
                c[tri] *= -1
                if self.correctQ:
                    c = self.q2_correction(c)
                c = c % self.Q

                assert (np.min(c) >= 0) and (np.max(c) < self.Q)

                if self.error:
                    e = np.int64(rng.normal(0, self.sigma, size=self.N).round())
                    b = (np.inner(c, self.secrets[idx]) + e) % self.Q
                else:
                    b = np.inner(c, self.secrets[idx]) % self.Q

                if self.correctQ:
                    b = self.q2_correction(b)

                return c, b
        if (self.rng.uniform() < self.maxQ_prob):
            maxQ = self.Q
        else: 
            maxQ = self.percQ_bound * self.Q

        if self.matrix_mode == "general":
            c = rng.randint(0, maxQ, size=(N, N), dtype=np.int64)
            c = c % self.Q
        else:
            # sample a uniformly from Z_q^n
            a = rng.randint(0, maxQ, size=N, dtype=np.int64)

            # do the circulant:
            c = circulant(a)
            tri = np.triu_indices(N, 1)
            c[tri] *= -1
            if self.correctQ:
                c = self.q2_correction(c)

            c = c % self.Q

        assert (np.min(c) >= 0) and (np.max(c) < self.Q)

        if self.error:
            e = np.int64(rng.normal(0, self.sigma, size = self.N).round())
            b = (c @ self.secrets[idx] + e) % self.Q
        else:
            b = (c @ self.secrets[idx]) % self.Q

        if self.correctQ:
            b = self.q2_correction(b)

        return c,b

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

    def get_difference(self, tgt, hyp):
        return abs(hyp[0]-tgt[0])

    def evaluate_bitwise(self, tgt, hyp):
        return [int(str(e1)==str(e2)) for e1,e2 in zip(tgt,hyp)]
