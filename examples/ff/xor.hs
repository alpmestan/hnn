module Main where

import AI.HNN.FF.Network
import Numeric.LinearAlgebra

samples :: Samples Double
samples = [ (fromList [0, 0], fromList [0])
          , (fromList [0, 1], fromList [1])
          , (fromList [1, 0], fromList [1])
          , (fromList [1, 1], fromList [0])
          ]

main :: IO ()
main = do
  n <- createNetwork 2 [2] 1
  mapM_ (print . output n tanh . fst) samples
  putStrLn "------------------"
  let n' = trainNTimes 1000 0.8 tanh tanh' n samples
  mapM_ (print . output n' tanh . fst) samples
  

{-

OUTPUT : 
[0.5835034660982111]
[0.7161238877902711]
[0.7476745582239942]
[0.7959844102953423]
------------------
[4.427877031872795e-2]
[0.9735228418859667]
[0.9544892359343744]
[-2.7925107071558753e-2]
-}
