{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}

-- |
-- Module       : AI.HNN.Recurrent.Network
-- Copyright    : (c) 2012 Gatlin Johnson
-- License      : BSD3
-- Maintainer   : rokenrol@gmail.com
-- Stability    : experimental
-- Portability  : GHC
--
-- An implementation of recurrent neural networks in pure Haskell.
--
-- A network is an adjacency matrix of connection weights, the number of
-- neurons, and the number of inputs.

module AI.HNN.Recurrent.Network (Network, createNetwork, computeStep,
                                 sigmoid, computeStepM) where

import AI.HNN.Internal.Matrix

import qualified Data.Vector                 as V
import qualified Data.Vector.Unboxed         as U
import qualified Data.Vector.Unboxed.Mutable as M

import System.Random.MWC
import Control.Monad

-- | Our recurrent neural network
data Network a = Network
                 { weights :: !(Matrix a)
                 , state   :: !(Vec a)
                 , size    :: {-# UNPACK #-} !Int
                 , nInputs :: {-# UNPACK #-} !Int
                 } deriving Show

-- | Creates a network with n neurons, m of which are inputs, and randomized weights
createNetwork :: (Variate a, U.Unbox a, Fractional a) => Int -> Int -> IO (Network a)
createNetwork n m = withSystemRandom . asGenST $ \gen -> do
    let ov = U.fromList (replicate n 0.0)
    rm <- uniformVector gen (n*n)
    return $! Network (Matrix rm n n) ov n m

-- | Evaluates a network with the specified function and list of inputs
--   precisely one time step.
computeStep :: (Variate a, U.Unbox a, Num a) =>
    Network a -> (a -> a) -> Vec a -> Vec a -> Network a

computeStep (Network{..}) activation thresh input =
    Network weights (overlay input next nInputs) size nInputs
    where
        overlay :: (Variate a, M.Unbox a) => Vec a -> Vec a -> Int -> Vec a
        overlay new old i = new U.++ (U.drop i old)
        {-# INLINE overlay #-}
        next = U.map activation $! U.zipWith (-) (weights `apply` state) thresh
        {-# INLINE next #-}

-- | Monadic version of computeStep, for convenience
computeStepM :: (Variate a, U.Unbox a, Num a, Monad m) =>
    Network a -> (a -> a) -> Vec a -> Vec a -> m (Network a)

computeStepM n a t i = return $ computeStep n a t i

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

