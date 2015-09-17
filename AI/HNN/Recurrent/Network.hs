{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}

-- |
-- Module       : AI.HNN.Recurrent.Network
-- Copyright    : (c) 2012 Gatlin Johnson
-- License      : LGPL
-- Maintainer   : rokenrol@gmail.com
-- Stability    : experimental
-- Portability  : GHC
--
-- An implementation of recurrent neural networks in pure Haskell.
--
-- A network is an adjacency matrix of connection weights, the number of
-- neurons, the number of inputs, and the threshold values for each neuron.
--
-- E.g.,
--
-- > module Main where
-- >
-- > import AI.HNN.Recurrent.Network
-- >
-- > main = do
-- >     let numNeurons = 3
-- >         numInputs  = 1
-- >         thresholds = replicate numNeurons 0.5
-- >         inputs     = [[0.38], [0.75]]
-- >         adj        = [ 0.0, 0.0, 0.0,
-- >                        0.9, 0.8, 0.0,
-- >                        0.0, 0.1, 0.0 ]
-- >     n <- createNetwork numNeurons numInputs adj thresholds :: IO (Network Double)
-- >     output <- evalNet n inputs sigmoid
-- >     putStrLn $ "Output: " ++ (show output)
--
-- This creates a network with three neurons (one of which is an input), an
-- arbitrary connection / weight matrix, and arbitrary thresholds for each neuron.
-- Then, we evaluate the network with an arbitrary input.
--
-- For the purposes of this library, the outputs returned are the values of all
-- the neurons except the inputs. Conceptually, in a recurrent net *any*
-- non-input neuron can be treated as an output, so we let you decide which
-- ones matter.

module AI.HNN.Recurrent.Network (

    -- * Network type
    Network, createNetwork,
    weights, size, nInputs, thresh,

    -- * Evaluation functions
    computeStep, evalNet,

    -- * Misc
    sigmoid

) where

import System.Random.MWC
import Control.Monad
import Numeric.LinearAlgebra hiding (i)
import Foreign.Storable as F

-- | Our recurrent neural network
data Network a = Network
                 { weights :: !(Matrix a)
                 , size    :: {-# UNPACK #-} !Int
                 , nInputs :: {-# UNPACK #-} !Int
                 , thresh  :: !(Vector a)
                 } deriving Show

-- | Creates a network with an adjacency matrix of your choosing, specified as
--   an unboxed vector. You also must supply a vector of threshold values.
createNetwork :: (Variate a, Fractional a, Storable a) =>
       Int            -- ^ number of total neurons neurons (input and otherwise)
    -> Int            -- ^ number of inputs
    -> [a]            -- ^ flat weight matrix
    -> [a]            -- ^ threshold (bias) values for each neuron
    -> IO (Network a) -- ^ a new network

createNetwork n m matrix thresh = return $!
    Network ( (n><n) matrix ) n m (n |> thresh)

-- | Evaluates a network with the specified function and list of inputs
--   precisely one time step. This is used by `evalNet` which is probably a
--   more convenient interface for client applications.
computeStep :: (Variate a, Num a, F.Storable a, Product a) =>
    Network a   -- ^ Network to evaluate input
    -> Vector a -- ^ vector of pre-existing state
    -> (a -> a) -- ^ activation function
    -> Vector a -- ^ list of inputs
    -> Vector a -- ^ new state vector

computeStep (Network{..}) state activation input =
    mapVector activation $! zipVectorWith (-) (weights <> prefixed) thresh
    where
        prefixed = Numeric.LinearAlgebra.vjoin
            [ input, (subVector nInputs (size-nInputs) state) ]
        {-# INLINE prefixed #-}

-- | Iterates over a list of input vectors in sequence and computes one time
--   step for each.
evalNet :: (Num a, Variate a, Fractional a, Product a) =>
    Network a        -- ^ Network to evaluate inputs
    -> [[a]]         -- ^ list of input lists
    -> (a -> a)      -- ^ activation function
    -> IO (Vector a) -- ^ output state vector

evalNet n@(Network{..}) inputs activation = do
    s <- foldM (\x -> computeStepM n x activation) state inputsV
    return $! subVector nInputs (size-nInputs) s
    where
        state = fromList $ replicate size 0.0
        {-# INLINE state #-}
        computeStepM _ s a i = return $ computeStep n s a i
        {-# INLINE computeStepM #-}
        inputsV = map (fromList) inputs
        {-# INLINE inputsV #-}

-- | It's a simple, differentiable sigmoid function.
sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}
