module Neuron (Neuron(..), compute, Sigmoid, Tanh, sigmoid, tanh) where
{-# LANGUAGE MultiParamTypeClasses, FlexibleContexts, FlexibleInstances, BangPatterns, EmptyDataDecls, RecordWildCards #-}

import Data.Vector.Unboxed (Vector, Unbox)
import qualified Data.Vector.Unboxed as V

class Floating a => Activation n a where
  activation_  :: n a -> (a -> a)
  activation'_ :: n a -> (a -> a)

-- compute :: Activation f => Neuron f a -> Vector a -> a
-- compute n inputs = activation

data Neuron f a = Neuron
                  { weights   :: Vector a
                  , threshold :: a
                  } deriving (Show)

data Sigmoid
data Tanh

instance Floating a => Activation (Neuron Sigmoid) a where
  activation_ _ = sigmoid
  {-# INLINE activation_ #-}

  activation'_ _ = sigmoid'
  {-# INLINE activation'_ #-}

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))

sigmoid' :: Floating a => a -> a
sigmoid' !x = case sigmoid x of
    s -> s * (1 - s)

{-# SPECIALIZE sigmoid :: Double -> Double #-}
{-# SPECIALIZE sigmoid :: Float  -> Float  #-}
--{-# INLINE sigmoid #-}

{-# SPECIALIZE sigmoid' :: Double -> Double #-}
{-# SPECIALIZE sigmoid' :: Float  -> Float  #-}
--{-# INLINE sigmoid' #-}

compute :: (Activation (Neuron f) a, Floating a, Unbox a)
        => Neuron f a -> Vector a -> a
compute n@(Neuron{..}) !inputs = activation_ n $ V.sum (V.zipWith (*) weights inputs)
{-# SPECIALIZE compute :: Activation (Neuron f) Double => Neuron f Double -> Vector Double -> Double #-}
{-# SPECIALIZE compute :: Activation (Neuron f) Float  => Neuron f Float  -> Vector Float  -> Float  #-}
--{-# INLINE compute #-}
