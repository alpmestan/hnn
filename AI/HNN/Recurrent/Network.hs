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
-- neurons, the number of inputs, and the threshold values for each neuron.
--
-- > module Main where
-- > import AI.HNN.Recurrent.Network
-- > import qualified Data.Vector.Unboxed as U
-- > import System.Random.MWC
-- >
-- > main = do
-- >     let numNeurons = 3
-- >         numInputs  = 2
-- >         thresholds = U.replicate numNeurons 0.5
-- >         input      = U.fromList [1, 0]
-- >
-- >     adj <- randomMatrix numNeurons
-- >     n <- createNetwork numNeurons numInputs adj thresholds :: IO (Network Double)
-- >     output <- evalNet n input sigmoid
-- >     putStrLn $ "Output: " ++ (show output)
-- >     where
-- >         randomMatrix n = withSystemRandom . asGenST $ \gen ->
-- >             uniformVector gen (n*n)
--
-- This creates a network with three neurons (two of which are inputs), a
-- random connection / weight matrix, and arbitrary thresholds for each neuron.
-- Then, we evaluate the network with an arbitrary input.
--
-- For the purposes of this library, the outputs returned are the values of all
-- the neurons except the inputs. Conceptually, in a recurrent net *any*
-- non-input neuron can be treated as an output, so we let you decide which
-- ones matter.

module AI.HNN.Recurrent.Network (Network, createNetwork, computeStep,
                                 sigmoid, evalNet, weights,
                                 size, nInputs, thresh) where

import AI.HNN.Internal.Matrix
import System.Random.MWC
import qualified Data.Vector.Unboxed as U
import Control.Monad

-- | Our recurrent neural network
data Network a = Network
                 { weights :: !(Matrix a)
                 , size    :: {-# UNPACK #-} !Int
                 , nInputs :: {-# UNPACK #-} !Int
                 , thresh  :: !(Vec a)
                 } deriving Show

-- | Creates a network with an adjacency matrix of your choosing, specified as
--   an unboxed vector. You also must supply a vector of threshold values.
createNetwork :: (Variate a, U.Unbox a, Fractional a) =>
    Int -> Int -> Vec a -> Vec a -> IO (Network a)

createNetwork n m matrix thresh = return $!
    Network (Matrix matrix n n) n m thresh

-- | Evaluates a network with the specified function and list of inputs
--   precisely one time step.
--
--   > netAfter = computeStep netBefore activation thresholdList inputs
--
--   A "threshold" for a neuron is a penalty deducted from the value
--   calculated. The thresholdList is a list of such for each neuron.
computeStep :: (Variate a, U.Unbox a, Num a) =>
    Network a -> Vec a -> (a -> a) -> Vec a -> Vec a

computeStep (Network{..}) state activation input =
    input U.++ (U.unsafeDrop nInputs next)
    where
        next = U.map activation $! U.zipWith (-) (weights `apply` state) thresh
        {-# INLINE next #-}

-- | Evaluate a network with a given input `n` timesteps, where n is the size
--   of the network.
evalNet :: (U.Unbox a, Num a, Variate a, Fractional a) =>
    Network a -> Vec a -> (a -> a) -> IO (Vec a)

evalNet n@(Network{..}) input activation = do
    s <- foldM (\x -> computeStepM n x activation) state $ replicate size input
    return $! U.unsafeDrop nInputs s
    where
        computeStepM n s a i = return $ computeStep n s a i
        {-# INLINE computeStepM #-}
        state = input U.++ (U.replicate (size - nInputs) 0.0)
        {-# INLINE state #-}

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}
