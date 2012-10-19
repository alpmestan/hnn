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

module AI.HNN.Recurrent.Network (Network, createNetwork, computeNetworkWith,
                                 sigmoid) where

import AI.HNN.Internal.Matrix

import qualified Data.Vector                 as V
import qualified Data.Vector.Unboxed         as U
import qualified Data.Vector.Unboxed.Mutable as M

import System.Random.MWC

-- | Our recurrent neural network
data Network a = Network
                 { weights :: !(Matrix a)
                 , size    :: {-# UNPACK #-} !Int
                 , nInputs :: {-# UNPACK #-} !Int
                 } deriving Show

-- | Creates a network with n neurons, m of which are inputs, and randomized weights
createNetwork :: (Variate a, U.Unbox a) => Int -> Int -> IO (Network a)
createNetwork n m = withSystemRandom . asGenST $ \gen -> do
    rm <- uniformVector gen (n*n)
    return $! Network (Matrix rm n n) n m

-- | Evaluates a network with the specified function and list of inputs
--   precisely one time step.
computeNetworkWith :: (U.Unbox a, Num a) =>
    Network a -> (a -> a) -> Vec a -> Vec a -> Vec a
computeNetworkWith (Network{..}) activation thresh input =
    U.map activation $! U.zipWith (-) (weights `apply` input) thresh

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

