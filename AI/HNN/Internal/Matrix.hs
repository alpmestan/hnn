{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}
module AI.HNN.Internal.Matrix (Matrix(..), Vec, getRow, getRows, dot, apply) where

import qualified Data.Vector.Unboxed as U

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
