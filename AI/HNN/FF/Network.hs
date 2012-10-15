{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}
module AI.HNN.FF.Network (Matrix, Network, Vec, createNetwork, computeNetworkWith, sigmoid, tanh) where

import qualified Data.Vector         as V
import qualified Data.Vector.Unboxed as U

import System.Random.MWC

-- | A matrix type based on unboxed vectors
data Matrix a = Matrix
                { mat  :: !(U.Vector a)
                , rows :: {-# UNPACK #-} !Int
                , cols :: {-# UNPACK #-} !Int
                } deriving Show

-- | Type for our vectors, synonym of 'Data.Vector.Unboxed.Vector a'
type Vec    a = U.Vector a

-- gets the nth row of a matrix
-- 0 <= n <= rows - 1
getRow :: U.Unbox a => Matrix a -> Int -> Vec a
getRow (Matrix{..}) n = U.unsafeSlice (n*cols) cols mat

-- gets the rows of the matrix as a list
getRows :: U.Unbox a => Matrix a -> [Vec a]
getRows m@(Matrix{..}) = map (getRow m) [0.. rows-1]

-- dot product
dot :: (U.Unbox a, Num a) => Vec a -> Vec a -> a
dot v1 v2 = U.sum $ U.zipWith (*) v1 v2

-- matrix mult
apply :: (U.Unbox a, Num a) => Matrix a -> Vec a -> Vec a
apply m@(Matrix{..}) v = let rs = getRows m in
  U.unfoldr go rs
  where go []     = Nothing
        go (r:rs) = Just (r `dot` v, rs)

-- | Our feed-forward neural network type
data Network a = Network
                 { matrices   :: !(V.Vector (Matrix a))
                 , thresholds :: !(V.Vector (Vec a))
                 , nInputs    :: {-# UNPACK #-} !Int
                 , arch       :: ![Int]
                 }

-- | `createNetwork n l` creates a neural network with 'n' inputs and if 'l' is [n1, n2, ...]
--   the net will have n1 neurons on the first layer, n2 neurons on the second, and so on
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
computeLayerWith :: (U.Unbox a, Num a) => (a -> a) -> Vec a -> (Matrix a, Vec a) -> Vec a
computeLayerWith f input (m, thresholds) = U.map f $! U.zipWith (-) (m `apply` input) thresholds 
{-# INLINE computeLayerWith #-}

-- | Computes the output of the given 'Network' assuming all neurons have the given function
--   as their activation function, and with input the given Vector
computeNetworkWith :: (U.Unbox a, Num a) => Network a -> (a -> a) -> Vec a -> Vec a
computeNetworkWith (Network{..}) activation input = V.foldl' (computeLayerWith activation) input $ V.zip matrices thresholds
{-# INLINE computeNetworkWith #-}

sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}