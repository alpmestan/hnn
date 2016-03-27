{-# LANGUAGE BangPatterns, 
             ScopedTypeVariables,
             RecordWildCards,
             FlexibleContexts,
             TypeFamilies,
             GeneralizedNewtypeDeriving #-}


module AI.HNN.FF.ComplexNetwork
    (
    -- * Types
      ComplexNetwork(..)
    , ActivationFunction
    , ActivationFunctionDerivative
    , Sample
    , Samples
    , (-->)

    -- * Creating a neural network
    , createComplexNetwork
    , fromWeightMatrices

    -- * Computing a neural network's output
    , output
    , sigmoid
    , sigmoid'
    , complexSigmoid
    , complexSigmoid'
    , randComplex

    -- * Training a neural network
    , trainUntil
    , trainNTimes
    , trainUntilErrorBelow
    , quadError
    
    -- * Loading and saving a neural network
    , loadComplexNetwork
    , saveComplexNetwork
    ) where

import Codec.Compression.Zlib     (compress, decompress)
import Data.Binary                (Binary(..), encode, decode)
import Data.List                  (foldl')
import Foreign.Storable           (Storable)
import qualified Data.ByteString.Lazy  as B
import qualified Data.Vector           as V
import qualified Data.Complex as C
import Debug.Trace

import System.Random.MWC
import Numeric.LinearAlgebra
import Data.Functor ((<$>))

-- | Our feed-forward neural network type. Note the 'Binary' instance, which means you can use 
--   'encode' and 'decode' in case you need to serialize your neural nets somewhere else than
--   in a file (e.g over the network)
newtype ComplexNetwork a = ComplexNetwork
                 { matrices   :: V.Vector (Matrix a) -- ^ the weight matrices
                 } deriving Show

instance (Element a, Binary a) => Binary (ComplexNetwork a) where
  put (ComplexNetwork ms) = put . V.toList $ ms
  get = (ComplexNetwork . V.fromList) `fmap` get                 


--instance (Complex a, Variate a) => Variate (Complex a) where
--	uniform g = uniform g :+ uniform g
--	uniformR (x,y) g = uniformR (x,y) g :+ uniformR (x,y) g




-- | The type of an activation function, mostly used for clarity in signatures
type ActivationFunction a = a -> a

-- | The type of an activation function's derivative, mostly used for clarity in signatures
type ActivationFunctionDerivative a = a -> a


-- 

randComplex :: (Variate a, Num a) => IO(Complex a)
randComplex = withSystemRandom . asGenST $ \gen -> do
  r <- uniformR (-1,1) gen
  i <- uniformR (-1,1) gen
  return (r :+ i)

randComplexList :: (Variate a, Num a) => Int -> IO([Complex a])
randComplexList 0 = return []
randComplexList n = do
  c <- randComplex
  lst <- randComplexList (n-1)
  return (c : lst)

randComplexMatrix :: (Variate a, Storable a, Num a) => (Int,Int) -> IO(Matrix (Complex a))
randComplexMatrix (rows,cols) = do
  lst <- randComplexList (rows*cols)
  return (reshape cols $ Numeric.LinearAlgebra.fromList lst)

-- | The following creates a neural network with 'n' inputs and if 'l' is [n1, n2, ...]
--   the net will have n1 neurons on the first layer, n2 neurons on the second, and so on
--   ending with k neurons on the output layer, with random weight matrices as a courtesy of
-- 'System.Random.MWC.uniformR'.
-- > createComplexNetwork n l k
createComplexNetwork :: (Variate a, Storable a, Num a) => Int -> [Int] -> Int -> IO (ComplexNetwork (Complex a))
createComplexNetwork nInputs hiddens nOutputs =
  fmap ComplexNetwork $ go dimensions V.empty
  where
        go [] !ms         = return ms
        go ((!n,!m):ds) ms = do
          !mat <- randComplexMatrix (n,m)
          go ds (ms `V.snoc` mat)
        dimensions      = zip (hiddens ++ [nOutputs]) $
                              (nInputs : hiddens)
{-# INLINE createComplexNetwork #-}



-- | Creates a neural network with exactly the weight matrices given as input here.
--   We don't check that the numbers of rows/columns are compatible, etc. 
fromWeightMatrices :: Storable a => V.Vector (Matrix (Complex a)) -> ComplexNetwork (Complex a)
fromWeightMatrices ws = ComplexNetwork ws
{-# INLINE fromWeightMatrices #-}

-- The `join [input, 1]' trick  below is a courtesy of Alberto Ruiz
-- <http://dis.um.es/~alberto/>. Per his words:
--
-- "The idea is that the constant input in the first layer can be automatically transferred to the following layers
-- by the learning algorithm (by setting the weights of a neuron to 1,0,0,0,...). This allows for a simpler
-- implementation and in my experiments those networks are able to easily solve non linearly separable problems."


--without the trick (of constant input)

output :: (Floating (Vector a), Product a, Storable a, Num (Vector a)) => ComplexNetwork a -> ActivationFunction a -> Vector a -> Vector a
output (ComplexNetwork{..}) act input = V.foldl' f input matrices
  where f !inp m = mapVector act $ m <> inp
{-# INLINE output #-}

-- | Computes and keeps the output of all the layers of the neural network with the given activation function
outputs :: (Floating (Vector a), Product a, Storable a, Num (Vector a)) => ComplexNetwork a -> ActivationFunction a -> Vector a -> V.Vector (Vector a)
outputs (ComplexNetwork{..}) act input = V.scanl f input matrices
  where f !inp m = mapVector act $ m <> inp
{-# INLINE outputs #-}


conju (x:+y) = x:+(-y)

deltas :: (Floating b, Floating (Vector a), Floating a, Product a, Storable a, Num (Vector a), Container Vector a, a ~ Complex b) => ComplexNetwork a -> ActivationFunctionDerivative a -> V.Vector (Vector a) -> Vector a -> V.Vector (Matrix a)
deltas (ComplexNetwork{..}) act' os expected = V.zipWith outer (V.tail ds) (V.init (V.map (mapVector conju) os))
  where !dl = (V.last os - expected) * (deriv $ mapVector conju (V.last os)) -- = (out_last - target)*(f'(netin_last)) = dev_last * deriv(out_last) = partial_last
        !ds = V.scanr f dl (V.zip os matrices) -- generates dev_i
        f (!o, m) !del = deriv o * ((ctrans m) <> del) -- dev_k = deriv(out_k)*dev_k = deriv(out_k)*(WEIGHTS <> partial_k+1)
        deriv = mapVector act'
{-# INLINE deltas #-}

updateComplexNetwork :: (Floating b, Floating (Vector a), Floating a, Product a, Storable a, Num (Vector a), Container Vector a, a ~ Complex b) => a -> ActivationFunction a -> ActivationFunctionDerivative a -> ComplexNetwork a -> Sample a -> ComplexNetwork a
updateComplexNetwork alpha act act' n@(ComplexNetwork{..}) (input, expectedOutput) = ComplexNetwork $ V.zipWith (+) matrices corr
    where !xs = outputs n act input
          !ds = deltas n act' xs expectedOutput
          !corr = V.map (scale (-alpha)) ds
{-# INLINE updateComplexNetwork #-}
          
-- | Input vector and expected output vector
type Sample a = (Vector a, Vector a)

-- | List of 'Sample's
type Samples a = [Sample a]

-- | Handy operator to describe your learning set, avoiding unnecessary parentheses. It's just a synonym for '(,)'. 
--   Generally you'll load your learning set from a file, a database or something like that, but it can be nice for 
--   quickly playing with hnn or for simple problems where you manually specify your learning set.
--   That is, instead of writing:
-- 
-- > samples :: Samples Double
-- > samples = [ (fromList [0, 0], fromList [0])
-- >           , (fromList [0, 1], fromList [1])
-- >           , (fromList [1, 0], fromList [1])
-- >           , (fromList [1, 1], fromList [0]) 
-- >           ]
-- 
--   You can write:
-- 
-- > samples :: Samples Double
-- > samples = [ fromList [0, 0] --> fromList [0]
-- >           , fromList [0, 1] --> fromList [1]
-- >           , fromList [1, 0] --> fromList [1]
-- >           , fromList [1, 1] --> fromList [0] 
-- >           ]
(-->) :: Vector a -> Vector a -> Sample a
(-->) = (,)

backpropOnce :: (Floating (Vector a), Floating b, Floating a, Product a, Num (Vector a), Container Vector a, a ~ Complex b) => a -> ActivationFunction a -> ActivationFunctionDerivative a -> ComplexNetwork a -> Samples a -> ComplexNetwork a
backpropOnce rate act act' n samples = foldl' (updateComplexNetwork rate act act') n samples
{-# INLINE backpropOnce #-}

-- | Generic training function.
-- 
-- The first argument is a predicate that will tell the backpropagation algorithm when to stop.
-- The first argument to the predicate is the epoch, i.e the number of times the backprop has been
-- executed on the samples. The second argument is /the current network/, and the third is the list of samples.
-- You can thus combine these arguments to create your own criterion.
-- 
-- For example, if you want to stop learning either when the network's quadratic error on the samples,
-- using the tanh function, is below 0.01, or after 1000 epochs, whichever comes first, you could
-- use the following predicate:
-- 
-- > pred epochs net samples = if epochs == 1000 then True else quadError tanh net samples < 0.01
-- 
-- You could even use 'Debug.Trace.trace' to print the error, to see how the error evolves while it's learning,
-- or redirect this to a file from your shell in order to generate a pretty graphics and what not.
-- 
-- The second argument (after the predicate) is the learning rate. Then come the activation function you want,
-- its derivative, the initial neural network, and your training set.
-- Note that we provide 'trainNTimes' and 'trainUntilErrorBelow' for common use cases.
trainUntil :: (Floating (Vector a), Floating b, Floating a, Product a, Num (Vector a), Container Vector a, a ~ Complex b) => (Int -> ComplexNetwork a -> Samples a -> Bool) -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> ComplexNetwork a -> Samples a -> ComplexNetwork a
trainUntil pr learningRate act act' net samples = go net 0
  where go n !k | pr k n samples = n
                | otherwise      = case backpropOnce learningRate act act' n samples of
                                    n' -> go n' (k+1)
{-# INLINE trainUntil #-}

-- | Trains the neural network with backpropagation the number of times specified by the 'Int' argument,
-- using the given learning rate (second argument).                                   
trainNTimes :: (Floating (Vector a), Floating b, Floating a, Product a, Num (Vector a), Container Vector a, a ~ Complex b) => Int -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> ComplexNetwork a -> Samples a -> ComplexNetwork a
trainNTimes n = trainUntil (\k _ _ -> k > n)
{-# INLINE trainNTimes #-}

-- | Quadratic error on the given training set using the given activation function. Useful to create
-- your own predicates for 'trainUntil'.
quadError :: (Floating (Vector a), Floating b, Floating a, Num (Vector a), Num (RealOf a), Product a, a ~ Complex b) => ActivationFunction a -> ComplexNetwork a -> Samples a -> RealOf a
quadError act net samples = foldl' (\err (inp, out) -> err + (norm2 $ output net act inp - out)) 0 samples
{-# INLINE quadError #-}

-- | Trains the neural network until the quadratic error ('quadError') comes below the given value (first argument),
-- using the given learning rate (second argument).
-- 
-- /Note/: this can loop pretty much forever when you're using a bad architecture for the problem, or unappropriate activation
-- functions.
trainUntilErrorBelow :: (Floating (Vector a), Floating b, Floating a, Product a, Num (Vector a), Ord a, Container Vector a, Num (RealOf a), a ~ RealOf a, Show a, a ~ Complex b) => a -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> ComplexNetwork a -> Samples a -> ComplexNetwork a
trainUntilErrorBelow x rate act = trainUntil (\_ n s -> quadError act n s < x) rate act
{-# INLINE trainUntilErrorBelow #-}

-- | The sigmoid function:  1 / (1 + exp (-x))
sigmoid :: Floating a => a -> a
sigmoid !x = 1 / (1 + exp (-x))
{-# INLINE sigmoid #-}

-- | Derivative of the sigmoid function: sigmoid x * (1 - sigmoid x)
sigmoid' :: Floating a => a -> a
sigmoid' !x = case sigmoid x of
  s -> s * (1 - s)
{-# INLINE sigmoid' #-}


complexSigmoid :: Floating a => Complex a -> Complex a
complexSigmoid (x :+ y) = (sigmoid x) :+ (sigmoid y)

complexSigmoid' :: Floating a => Complex a -> Complex a
complexSigmoid' (x :+ y) = (sigmoid' x) :+ (sigmoid' y)

-- | Loading a neural network from a file (uses zlib compression on top of serialization using the binary package).
--   Will throw an exception if the file isn't there.
loadComplexNetwork :: (Storable a, Element a, Binary a) => FilePath -> IO (ComplexNetwork a)
loadComplexNetwork fp = decode . decompress <$> B.readFile fp
{-# INLINE loadComplexNetwork #-}

-- | Saving a neural network to a file (uses zlib compression on top of serialization using the binary package).
saveComplexNetwork :: (Storable a, Element a, Binary a) => FilePath -> ComplexNetwork a -> IO ()
saveComplexNetwork fp = B.writeFile fp . compress . encode
{-# INLINE saveComplexNetwork #-}
