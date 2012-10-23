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
-- >        inputs     = map (U.fromList) [ [1, 0]
-- >                                      , [0, 1]
-- >                                      , [2, 3]
-- >                                      ]
-- >        adjmatrix  = [ 0.5, 0.3, 0.9,
-- >                       0.1, 0.8, 0.4,
-- >                       0.7, 0.6, 0.2 ]
-- >    n <- createNetworkWith numNeurons numInputs (U.fromList adjmatrix)
-- >    n <- foldM (\x -> computeStepM x sigmoid thresholds) n inputs
-- >    putStrLn . show $ output n 1 -- get 1 output
--
-- This example creates a network with 3 neurons, the "first" two of which are
-- input neurons, and steps it over 3 input vectors.
--
-- In a recurrent network, *any* non-input value can be usefully considered an
-- output. By convention, then, calling `output net n` returns a vector of the
-- first `n` neuron values after the inputs. Think of it like this:
--
-- > [ input1, input2, output1, ..., outputN]
--
-- It is up to you to structure your net accordingly. The upcoming
-- neuro-evolution training algorithm will also follow this convention.

module AI.HNN.Recurrent.Network (Network, createNetwork, computeStep,
                                 sigmoid, computeStepM, weights,
                                 state, size, nInputs, output, createNetworkWith) where

import AI.HNN.Internal.Matrix
import System.Random.MWC

import qualified Data.Vector.Unboxed as U


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
    return $! Network (Matrix rm n n) (U.replicate n 0.0) n m

-- | Creates a network with an adjacency matrix of your choosing, specified as
--   an unboxed vector.
createNetworkWith :: (Variate a, U.Unbox a, Fractional a) => Int -> Int -> Vec a ->
    IO (Network a)
createNetworkWith n m matrix = return $! Network (Matrix matrix n n)
    (U.replicate n 0.0) n m

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

-- | Grab *n* outputs from the net.
output :: (U.Unbox a) => Network a -> Int -> Vec a
output (Network{..}) n = U.unsafeSlice nInputs n state
{-# INLINE output #-}

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

