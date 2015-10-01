-- might not work if module is not installed, possible fix is to place the file in 
-- hnn-master root folder

import AI.HNN.FF.ComplexNetwork
import Numeric.LinearAlgebra
import Data.Complex
import System.Random.MWC
import Foreign.Storable           (Storable)


samplesXOR :: Samples (Complex Double)
samplesXOR = [ (fromList [(-1):+ (-1)], fromList [1:+0])
          , (fromList [(-1):+1], fromList [0:+0])
          , (fromList [1:+(-1)], fromList [1:+1])
          , (fromList [1:+1], fromList [0:+1])
          ]

real' :: (Floating a) => Complex a -> Complex a
real' (x :+ y) = x :+ 0

mapComplex ::  (Complex a -> a) -> (Complex a -> a) -> Complex a -> Complex a
mapComplex f g z = (f z) :+ (g z)

quarters :: (Floating a, Ord a, RealFloat a) => Complex a -> Complex a
quarters (x :+ y) = (if x > 0 then 1 else 0) :+ (if y > 0 then 1 else 0)

makeSamples :: (Floating a, Variate a, Storable a) => Int -> (Complex a -> Complex a) -> IO (Samples (Complex a))
makeSamples 0 _ = return []
makeSamples n f = do
  z <- randComplex
  lst <- makeSamples (n-1) f
  return ((fromList [z], fromList [f(z)]) : lst)


-- uncomment relevant sections for different tests
main :: IO ()
main = do
  -- Complex XOR:
  --mapM_ (putStrLn . show ) samplesXOR
  --n <- createComplexNetwork 1 [2] 1 :: IO (ComplexNetwork (Complex Double))
  --putStrLn $ show n
  --putStrLn "------------------"
  --let n' = trainNTimes 1000 0.8 complexSigmoid complexSigmoid' n samplesXOR
  --mapM_ (putStrLn . show . output n' complexSigmoid . fst) samplesXOR
  --putStrLn "------------"
  --putStrLn $ show n'

  -- Quarters on plane:
  --samples <- makeSamples 50 quarters
  --test <- makeSamples 10 quarters
  --mapM_ (putStrLn . show ) test
  --n <- createComplexNetwork 1 [2] 1 :: IO (ComplexNetwork (Complex Double))
  --putStrLn $ show n
  --putStrLn "------------------"
  --let n' = trainNTimes 1000 0.8 complexSigmoid complexSigmoid' n samples
  --mapM_ (putStrLn . show . output n' complexSigmoid . fst) test
  --putStrLn "------------"
  --putStrLn $ show n'

  -- Parabolas on plane: 
  let h1 (x :+ y) = if x > 0.5*x^2 -0.5 then 1 else 0
  let h2 (x :+ y) = if y > -0.5*x^2 +0.5 then 1 else 0
  
  samples <- makeSamples 1000 (mapComplex h1 h2)
  test <- makeSamples 10 (mapComplex h1 h2 )

  mapM_ (putStrLn . show ) test
  n <- createComplexNetwork 1 [20] 1 :: IO (ComplexNetwork (Complex Double))
  let n' = trainNTimes 1 0.5 complexSigmoid complexSigmoid' n samples

  mapM_ (putStrLn . show . output n' complexSigmoid . fst) test
