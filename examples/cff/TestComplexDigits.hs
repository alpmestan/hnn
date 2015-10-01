-- might not work if module is not installed, possible fix is to place the file in 
-- hnn-master root folder

import AI.HNN.FF.ComplexNetwork
import Numeric.LinearAlgebra
import Data.Complex
import System.Random.MWC
import Foreign.Storable           (Storable)


-- actually ignoring imaginary part, adding constant +0i to all real numbers
samples :: Samples (Complex Double)
samples = [
  (fromList [ 1:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 0:+0, 0:+0, 1:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 1:+0, 1:+0, 1:+0, 1:+0 ], fromList [1:+0]),   -- three
  
  (fromList [ 1:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 1:+0, 1:+0, 1:+0, 1:+0 ], fromList [1:+0]),   -- three
  
  (fromList [ 0:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 1:+0, 1:+0
            , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 0:+0, 1:+0, 1:+0, 1:+0, 1:+0 ], fromList [1:+0]),   -- three
  
  (fromList [ 1:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 1:+0
            , 1:+0, 1:+0, 1:+0, 1:+0, 1:+0 ], fromList [0:+0]),   -- not a three
  
  (fromList [ 1:+0, 1:+0, 1:+0, 1:+0, 1:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 0:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 0:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 0:+0
            , 1:+0, 0:+0, 0:+0, 0:+0, 0:+0 ], fromList [0:+0]),   -- not a three

  (fromList [ 0:+0, 1:+0, 1:+0, 1:+0, 0:+0
            , 0:+0, 1:+0, 0:+0, 1:+0, 0:+0
            , 0:+0, 1:+0, 1:+0, 1:+0, 0:+0
            , 0:+0, 1:+0, 0:+0, 1:+0, 0:+0
            , 0:+0, 1:+0, 1:+0, 1:+0, 0:+0 ], fromList [0:+0]),   -- not a three

  (fromList [ 0:+0, 0:+0, 1:+0, 0:+0, 0:+0
            , 0:+0, 1:+0, 1:+0, 0:+0, 0:+0
            , 1:+0, 0:+0, 1:+0, 0:+0, 0:+0
            , 0:+0, 0:+0, 1:+0, 0:+0, 0:+0
            , 0:+0, 0:+0, 1:+0, 0:+0, 0:+0 ], fromList [0:+0]) ]  -- not a three


main :: IO ()
main = do
  n <- createComplexNetwork 25 [250] 1
  let n' = trainNTimes 100 0.5 complexSigmoid complexSigmoid' n samples
  mapM_ (putStrLn . show . output n' complexSigmoid . fst) samples
  putStrLn "-------------"
  putStrLn . show . output n' complexSigmoid $ testInput

  where testInput = fromList [ 0:+0, 0:+0, 1:+0, 1:+0, 1:+0
                             , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
                             , 0:+0, 0:+0, 1:+0, 1:+0, 1:+0
                             , 0:+0, 0:+0, 0:+0, 0:+0, 1:+0
                             , 0:+0, 0:+0, 1:+0, 1:+0, 1:+0 ]

