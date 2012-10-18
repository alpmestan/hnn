{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards #-}
module AI.HNN.Internal.Matrix (Matrix(..), Vec, getRow, getRows, dot, apply, (*:), (^:), applyTransposed, matMap, matAdd, printM) where

import qualified Data.Vector.Unboxed as U


-- | A matrix type based on unboxed vectors
data Matrix a = Matrix
                { mat   :: !(U.Vector a)
                , rows  :: {-# UNPACK #-} !Int
                , cols  :: {-# UNPACK #-} !Int
                }
matMap :: U.Unbox a => (a -> a) -> Matrix a -> Matrix a
matMap f (Matrix{..}) = Matrix (U.map f mat) rows cols

matAdd :: (U.Unbox a, Num a) => Matrix a -> Matrix a -> Matrix a
matAdd m1 m2 = Matrix (U.zipWith (+) (mat m1) (mat m2)) (rows m1) (cols m1)

swap :: (a, b) -> (b, a)
swap (a,b) = (b,a)

indexToXY :: (Int, Int) -> Int -> (Int, Int)
indexToXY (_, c) i = (i `quot` c, i `rem` c)

xyToIndex :: (Int, Int) -> (Int, Int) -> Int
xyToIndex (_,c) (x,y) = x * c + y

-- THIS IS BAD. WE MUST IMPROVE THIS. xyToIndex should be defined in the 'where' to avoid (rows, cols) as arg
transpose :: U.Unbox a => Matrix a -> Matrix a
transpose (Matrix{..}) = Matrix mat' cols rows
  where mat' = U.unfoldr go 0
        go i | i == rows*cols = Nothing
             | otherwise      = let ti = xyToIndex (rows, cols) . swap . indexToXY (cols, rows) $! i in Just (mat U.! ti, i+1)

instance (Show a, U.Unbox a) => Show (Matrix a) where
  show m@(Matrix{..}) = "\n---------Matrix--------\n" ++ show rows ++ "x" ++ show cols ++ "\n" ++ show (getRows m)

printM :: (U.Unbox a, Show a) => Matrix a -> IO ()
printM m@(Matrix{..}) = do
  putStrLn $ show rows ++ "x" ++ show cols
  mapM_ (putStrLn . show) $ getRows m
  
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

-- matrix-vector mult
apply :: (U.Unbox a, Num a) => Matrix a -> Vec a -> Vec a
apply m@(Matrix{..}) v = let rs = getRows m in
  U.unfoldr go rs
  where go []     = Nothing
        go (r:rs) = Just (r `dot` v, rs)

-- m `applyTransposed` x = m^T `apply` x
applyTransposed :: (U.Unbox a, Num a) => Matrix a -> Vec a -> Vec a
applyTransposed m@(Matrix{..}) x = transpose m `apply` x

-- u *: v = (u_i * v_i)_i
(*:) :: (U.Unbox a, Num a) => Vec a -> Vec a -> Vec a
u *: v = U.zipWith (*) u v

-- u (^:) v = (u_i * v_j)_i,j
(^:) :: (U.Unbox a, Num a) => Vec a -> Vec a -> Matrix a
u ^: v = Matrix mat (U.length u) (U.length v)
  where mat = U.concatMap (\u_i -> U.map (*u_i) v) u
