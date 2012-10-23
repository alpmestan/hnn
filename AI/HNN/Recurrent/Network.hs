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
--
-- Usage:
--
-- > main = do
-- >    let numNeurons = 3
-- >        numInputs  = 2
-- >        thresholds = U.fromList $ replicate numNeurons 0.5
-- >        inputs     = map (U.fromList) [ [1, 1]
-- >                                      , [2, 3]
-- >                                      , [1, 3]
-- >                                      , [0, 2]
-- >                                      ]
-- >    n <- createNetwork numNeurons numInputs
-- >    n <- foldM (\x -> computeStepM x sigmoid thresholds) n inputs
-- >    putStrLn . show $ n
--
-- This would create a network with *randomized* connections of *randomized*
-- weights among neurons. Then this trivial example runs through 4 steps of
-- feeding inputs into the network and computing the next state.

module AI.HNN.Recurrent.Network (Network, createNetwork, computeStep,
                                 sigmoid, computeStepM, weights,
                                 state, size, nInputs) where

import AI.HNN.Internal.Matrix

import qualified Data.Vector                 as V
import qualified Data.Vector.Unboxed         as U

import System.Random.MWC

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
    rm <- uniformVector gen (n*n)
    return $! Network (Matrix rm n n) ov n m
    where
        ov :: (Fractional a, U.Unbox a) => Vec a
        ov = U.replicate n 0.0
        {-# INLINE ov #-}

-- | Evaluates a network with the specified function and list of inputs
--   precisely one time step.
--
--   > netAfter = computeStep netBefore activation thresholdList inputs
--
--   A "threshold" for a neuron is a penalty deducted from the value
--   calculated. The thresholdList is a list of such for each neuron.
computeStep :: (Variate a, U.Unbox a, Num a) =>
    Network a -> (a -> a) -> Vec a -> Vec a -> Network a

computeStep (Network{..}) activation thresh input =
    Network weights (overlay input next nInputs) size nInputs
    where
        overlay :: (Variate a, U.Unbox a) => Vec a -> Vec a -> Int -> Vec a
        overlay new old i = new U.++ (U.unsafeDrop i old)
        {-# INLINE overlay #-}
        next = U.map activation $! U.zipWith (-) (weights `apply` state) thresh
        {-# INLINE next #-}

-- | Monadic version of `computeStep`, for convenience.
computeStepM :: (Variate a, U.Unbox a, Num a, Monad m) =>
    Network a -> (a -> a) -> Vec a -> Vec a -> m (Network a)

computeStepM n a t i = return $ computeStep n a t i
{-# INLINE computeStepM #-}

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

