{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}

-- |
-- Module       : AI.HNN.FF.Network
-- Copyright    : (c) 2012 Alp Mestanogullari
-- License      : BSD3
-- Maintainer   : alpmestan@gmail.com
-- Stability    : experimental
-- Portability  : GHC
-- 
-- An implementation of feed-forward neural networks in pure Haskell.
-- 
-- It uses weight matrices between each layer to represent the connections between neurons from
-- a layer to the next and exports only the useful bits for a user of the library.
-- 
-- Here is an example of using this module to create a feed-forward neural network with 2 inputs,
-- 2 neurons in a hidden layer and one neuron in the output layer, with random weights, and compute
-- its output for [1,2] using the sigmoid function for activation for all the neurons.
-- 
-- > import AI.HNN.FF.Network
-- > import qualified Data.Vector.Unboxed as U
-- >
-- > main = do
-- >   n <- createNetwork 2 [2, 1] :: IO (Network Double)
-- >   print $ computeNetworkWith n sigmoid (U.fromList [0.5, 0.5])
-- 
-- /Note/: Here, I create a @Network Double@, but you can replace 'Double' with any number type
-- that implements the @System.Random.MWC.Variate@, @Num@ and @Data.Vector.Unboxed.Unbox@
-- typeclasses. Having your number type implement the @Floating@ typeclass too is a good idea, since that's what most of the
-- common activation functions require.

module AI.HNN.FF.Network (Network, Vec, createNetwork, computeNetworkWith, computeNetworkWithS, sigmoid, tanh) where

import qualified Data.Vector         as V
import qualified Data.Vector.Unboxed as U

import System.Random.MWC

import AI.HNN.Internal.Matrix

-- | Our feed-forward neural network type
data Network a = Network
                 { matrices   :: !(V.Vector (Matrix a))
                 , thresholds :: !(V.Vector (Vec a))
                 , nInputs    :: {-# UNPACK #-} !Int
                 , arch       :: ![Int]
                 }

-- | The following creates a neural network with 'n' inputs and if 'l' is [n1, n2, ...]
--   the net will have n1 neurons on the first layer, n2 neurons on the second, and so on
-- 
-- > createNetwork n l
createNetwork :: (Variate a, U.Unbox a) => Int -> [Int] -> IO (Network a)
createNetwork nI as = withSystemRandom . asGenST $ \gen -> do
  (vs, ts) <- go nI as V.empty V.empty gen
  return $! Network vs ts nI as
  where go _  []         ms ts _ = return $! (ms, ts)
        go !k (!a:archs) ms ts g = do
          m  <- randomMatrix a k g
          let !m' = Matrix m a k
          t  <- randomMatrix a 1 g
          go a archs (ms `V.snoc` m') (ts `V.snoc` t) g

        randomMatrix n m g = uniformVector g (n*m)

-- Helper function that computes the output of a given layer
computeLayerWith :: (U.Unbox a, Num a) => Vec a -> (Matrix a, Vec a, a -> a) -> Vec a
computeLayerWith input (m, thresholds, f) = U.map f $! U.zipWith (-) (m `apply` input) thresholds 
{-# INLINE computeLayerWith #-}

-- | Computes the output of the given 'Network' assuming all neurons have the given function
--   as their activation function, and with input the given 'Vec'
-- 
-- Example:
-- 
-- > computeNetworkWith n sigmoid (U.fromList [0.5, 0.5])
computeNetworkWith :: (U.Unbox a, Num a) => Network a -> (a -> a) -> Vec a -> Vec a
computeNetworkWith (Network{..}) activation input = V.foldl' computeLayerWith input $ V.zip3 matrices thresholds (V.replicate (length arch) activation)
{-# INLINE computeNetworkWith #-}

-- | Computes the output of the given 'Network', just like 'computeNetworkWith', but accepting
--   different activation functions on each layer. We thus have:
-- 
-- > computeNetworkWith n f input == computeNetworkWithS n (repeat f) input
-- 
-- (or, to be more accurate, we can replace @repeat f@ by a list containing a copy of @f@ per layer)
computeNetworkWithS :: (U.Unbox a, Num a) => Network a -> [a -> a] -> Vec a -> Vec a
computeNetworkWithS (Network{..}) activations input = V.foldl' computeLayerWith input $ V.zip3 matrices thresholds (V.fromList activations)

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}
