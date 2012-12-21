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
  mapM_ (putStrLn . show . output n tanh . fst) samples
  putStrLn "------------------"
  let n' = trainNTimes 1000 0.8 tanh tanh' n samples
  mapM_ (putStrLn . show . output n' tanh . fst) samples
  