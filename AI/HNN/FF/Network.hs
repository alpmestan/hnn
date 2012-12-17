{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}

-- |
-- Module       : AI.HNN.FF.Network
-- Copyright    : (c) 2012 Alp Mestanogullari, Gatlin Johnson
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

module AI.HNN.FF.Network (

    -- * Network type
    Network,
    createNetwork,
    computeNetworkWith,
    --computeNetworkWithS,

    -- * Utilities
    sigmoid,

    -- * Training
    backprop

) where

import System.Random.MWC
import Numeric.LinearAlgebra
import Foreign.Storable as F
import Data.List (scanl)

-- | Our feed-forward neural network type
data Network a = Network
                 { matrices   :: !(Vector (Matrix a))
                 , thresholds :: !(Vector (Vector a))
                 , nInputs    :: {-# UNPACK #-} !Int
                 , arch       :: ![Int]
                 }

-- | The following creates a neural network with 'n' inputs and if 'l' is [n1, n2, ...]
--   the net will have n1 neurons on the first layer, n2 neurons on the second, and so on
-- 
-- > createNetwork n l
createNetwork :: (Variate a, Fractional a, Storable a) =>
    Int   ->       -- ^ Number of input neurnos
    [Int] ->       -- ^ List of number of neurons for remaining layers
    IO (Network a)

createNetwork nI as = do
    (ms, ts) <- initalValues nI as [] []
    return $! Network ms ts nI as
    where
        initialValues _ [] ms ts          = return $! (fromList ms, fromList ts)
        initialValues !k (!a:archs) ms ts = do
            m <- rand a k
            t <- rand a 1
            initialValues a archs (ms ++ [m]) (ts ++ [t])
        empty = fromList []

-- Helper function that computes the output of a given layer
computeLayerWith :: (Variate a, Num a, F.Storable a, Product a) =>
    (a -> a)              -> -- ^ activation function
    (Matrix a, Vector a)  -> -- ^ Matrix and associated thresholds
    Vector a              -> -- ^ input vector
    Vector a

computeLayerWith f (m, thresholds) input =
    mapVector f $! zipVectorWith (-) (m <> input) thresholds
{-# INLINE computeLayerWith #-}

-- | Computes the output of the given 'Network' assuming all neurons have the given function
--   as their activation function, and with input the given Vector
-- 
-- Example:
-- 
-- > computeNetworkWith n sigmoid (U.fromList [0.5, 0.5])
computeNetworkWith :: (Variate a, Num a, Fractional a, Product a) =>
    Network a    -> -- * the network
    (a -> a)     -> -- * activation
    Vector a     -> -- * input
    Vector a

computeNetworkWith (Network{..}) activation input =
    foldVectorL (computeLayerWith activation) input $!
        zipVector matrices thresholds
    where
        foldVectorL :: (Storable a) => (a -> b -> a) -> a -> [b] -> a
        foldVectorL f a bs =
            foldVector (\b g x -> g (f x b) ) id bs a
        {-# INLINE foldVectorL #-}
{-# INLINE computeNetworkWith #-}

-- | Computes the output of the given 'Network', just like 'computeNetworkWith', but accepting
--   different activation functions on each layer. We thus have:
-- 
-- > computeNetworkWith n f input == computeNetworkWithS n (repeat f) input
-- 
-- (or, to be more accurate, we can replace @repeat f@ by a list containing a copy of @f@ per layer)
{- computeNetworkWithS :: (Variate a, Num a, Fractional a, Product a) =>
    Network a ->
    [a -> a]  ->
    Vec a     ->
    Vec a
computeNetworkWithS (Network{..}) activations input =
    V.foldl' computeLayerWith input $
        V.zip3 matrices thresholds (V.fromList activations) -}

-- | Simple implementation of backpropagation for fully connected FF networks
backprop :: (Variate a, Num a, Fractional a, Product a) =>
    Network a              -> -- * Network to train
    (a -> a)               -> -- * activation function
    [(Vector a, Vector a)] -> -- * training set
    IO (Network a)

backprop net f training =
    foldM bp net training
    where
        bp n@(Network{..}) (x,t) = do
            let ys  = fromList $!
                    scanl (computeLayerWith f) x $! zipVector matrices thresholds
                dlt = zipVectorWith (-) t $! lastV ys
            -- 
            return $! Network mts ths nInputs arch
        where
            lastV vs = vs @> (dim vs)-1
            {-# INLINE lastV #-}

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}
