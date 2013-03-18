{-# LANGUAGE BangPatterns, ScopedTypeVariables, RecordWildCards, FlexibleContexts, TypeFamilies #-}

-- |
-- Module       : AI.HNN.FF.Network
-- Copyright    : (c) 2012 Alp Mestanogullari
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
-- > import Numeric.LinearAlgebra
-- >
-- > main = do
-- >   n <- createNetwork 2 [2] 1 :: IO (Network Double)
-- >   print $ output n sigmoid (fromList [1, 1])
-- 
-- /Note/: Here, I create a @Network Double@, but you can replace 'Double' with any number type
-- that implements the appropriate typeclasses you can see in the signatures of this module.
-- Having your number type implement the @Floating@ typeclass too is a good idea, since that's what most of the
-- common activation functions require.
--
-- /Note 2/: You can also give some precise weights to initialize the neural network with, with
-- 'fromWeightMatrices'. You can also restore a neural network you had saved using 'loadNetwork'.
-- 
-- Here is an example of how to train a neural network to learn the XOR function.
-- ( for reference: XOR(0, 0) = 0, XOR(0, 1) = 1, XOR(1, 0) = 1, XOR(1, 1) = 0 )
-- 
-- First, let's import hnn's feedforward neural net module, and hmatrix's vector types.
-- 
-- > import AI.HNN.FF.Network
-- > import Numeric.LinearAlgebra
-- 
-- Now, we will specify our training set (what the net should try to learn).
-- 
-- > samples :: Samples Double
-- > samples = [ (fromList [0, 0], fromList [0])
-- >           , (fromList [0, 1], fromList [1])
-- >           , (fromList [1, 0], fromList [1])
-- >           , (fromList [1, 1], fromList [0]) ]
-- 
-- You can see that this is basically a list of pairs of vectors, the first vector being
-- the input given to the network, the second one being the expected output. Of course,
-- this imply working on a neural network with 2 inputs, and a single neuron on the output layer. Then,
-- let's create one!
-- 
-- > main = do
-- >   net <- createNetwork 2 [2] 1
-- 
-- You may have noticed we haven't specified a signature this time, unlike in the earlier snippet.
-- Since we gave a signature to samples, specifying we're working with 'Double' numbers, and since
-- we are going to tie 'net' and 'samples' by a call to a learning function, GHC will gladly figure out
-- that 'net' is working with 'Double'. 
-- 
-- Now, it's time to train our champion. But first, let's see how bad he is now. The weights are most likely
-- not close to those that will give a good result for simulating XOR. Let's compute the output of the net on
-- the input vectors of our samples, using 'tanh' as the activation function.
-- 
-- >   mapM_ (print . output net tanh . fst) samples
-- 
-- Ok, you've tested this, and it gives terrible results. Let's fix this by letting 'trainNTimes' teach our neural net
-- how to behave. Since we're using 'tanh' as our activation function, we will tell it to the training function,
-- and also specify its derivative.
-- 
-- >   let smartNet = trainNTimes 1000 0.8 tanh tanh' net samples
-- 
-- So, this tiny piece of code will run the backpropagation algorithm on the samples 1000 times, with a learning rate
-- of 0.8. The learning rate is basically how strongly we should modify the weights when we try to correct the error the net makes
-- on our samples. The bigger it is, the more the weights are going to change significantly. Depending on the cases, it is good,
-- but sometimes it can also make the backprop algorithm oscillate around good weight values without actually getting to them.
-- You usually want to test several values and see which ones gets you the nicest neural net, which generalizes well to samples
-- that are not in the training set while giving decent results on the training set.
-- 
-- Now, let's see how that worked out for us:
-- 
-- >   mapM_ (print . output smartNet tanh . fst) samples
-- 
-- You could even save that neural network's weights to a file, so that you don't need to train it again in the future, using 'saveNetwork':
--
-- >   saveNetwork "smartNet.nn" smartNet
--
-- Please note that 'saveNetwork' is just a wrapper around zlib compression + serialization using the binary package.
-- AI.HNN.FF.Network also provides a 'Data.Binary.Binary' instance for 'Network', which means you can also simply use
-- 'Data.Binary.encode' and 'Data.Binary.decode' to have your own saving/restoring routines, or to simply get a bytestring 
-- we can send over the network, for example.
-- 
-- Here's a run of the program we described on my machine (with the timing): first set of
-- fromList's is the output of the initial neural network, the second one is the output of
-- 'smartNet' :-)
-- 
-- > fromList [0.574915179613429]
-- > fromList [0.767589097192215]
-- > fromList [0.7277396607146663]
-- > fromList [0.8227114080561128]
-- > ------------------
-- > fromList [6.763498312099933e-2]
-- > fromList [0.9775186355284375]
-- > fromList [0.9350823296850516]
-- > fromList [-4.400205702560454e-2]
-- > 
-- > real    0m0.365s
-- > user    0m0.072s
-- > sys     0m0.016s
-- 
-- Rejoyce! Feel free to play around with the library and report any bug, feature request and whatnot to us on
-- our github repository <https://github.com/alpmestan/hnn/issues> using the appropriate tags. Also, you can
-- see the simple program we studied here with pretty colors at <https://github.com/alpmestan/hnn/blob/master/examples/ff/xor.hs>
-- and other ones at <https://github.com/alpmestan/hnn/tree/master/examples/ff>.

module AI.HNN.FF.Network
    (
    -- * Types
      Network(..)
    , ActivationFunction
    , ActivationFunctionDerivative
    , Sample
    , Samples

    -- * Creating a neural network
    , createNetwork
    , fromWeightMatrices

    -- * Computing a neural network's output
    , output
    , tanh
    , tanh'
    , sigmoid
    , sigmoid'

    -- * Training a neural network
    , trainUntil
    , trainNTimes
    , trainUntilErrorBelow
    , quadError
    
    -- * Loading and saving a neural network
    , loadNetwork
    , saveNetwork
    ) where

import Codec.Compression.Zlib     (compress, decompress)
import Data.Binary                (Binary(..), encode, decode)
import Data.Vector.Binary         ()
import Data.List                  (foldl')
import Foreign.Storable           (Storable)
import qualified Data.ByteString.Lazy  as B
import qualified Data.Vector           as V

import System.Random.MWC
import Numeric.LinearAlgebra

-- | Our feed-forward neural network type. Note the 'Binary' instance, which means you can use 
--   'encode' and 'decode' in case you need to serialize your neural nets somewhere else than
--   in a file (e.g over the network)
newtype Network a = Network
                 { matrices   :: V.Vector (Matrix a) -- ^ the weight matrices
                 } deriving (Show)

instance (Element a, Binary a) => Binary (Network a) where
  put (Network ms) = put ms
  get = Network `fmap` get                 

-- | The type of an activation function, mostly used for clarity in signatures
type ActivationFunction a = a -> a

-- | The type of an activation function's derivative, mostly used for clarity in signatures
type ActivationFunctionDerivative a = a -> a

-- | The following creates a neural network with 'n' inputs and if 'l' is [n1, n2, ...]
--   the net will have n1 neurons on the first layer, n2 neurons on the second, and so on
--   ending with k neurons on the output layer, with random weight matrices as a courtesy of
-- 'System.Random.MWC.uniformVector'.
-- 
-- > createNetwork n l k
createNetwork :: (Variate a, Storable a) => Int -> [Int] -> Int -> IO (Network a)
createNetwork nInputs hiddens nOutputs =
  fmap Network $ withSystemRandom . asGenST $ \gen -> go gen dimensions V.empty
  where
        go _ [] !ms         = return ms
        go gen ((!n,!m):ds) ms = do
          !mat <- randomMat n m gen
          go gen ds (ms `V.snoc` mat)
        randomMat n m g = reshape m `fmap` uniformVector g (n*m)
        dimensions      = zip (hiddens ++ [nOutputs]) $
                              (nInputs+1 : hiddens)
{-# INLINE createNetwork #-}


-- | Creates a neural network with exactly the weight matrices given as input here.
--   We don't check that the numbers of rows/columns are compatible, etc. 
fromWeightMatrices :: Storable a => V.Vector (Matrix a) -> Network a
fromWeightMatrices ws = Network ws
{-# INLINE fromWeightMatrices #-}

-- The `join [input, 1]' trick  below is a courtesy of Alberto Ruiz
-- <http://dis.um.es/~alberto/>. Per his words:
--
-- "The idea is that the constant input in the first layer can be automatically transferred to the following layers
-- by the learning algorithm (by setting the weights of a neuron to 1,0,0,0,...). This allows for a simpler
-- implementation and in my experiments those networks are able to easily solve non linearly separable problems."

-- | Computes the output of the network on the given input vector with the given activation function
output :: (Floating (Vector a), Product a, Storable a, Num (Vector a)) => Network a -> ActivationFunction a -> Vector a -> Vector a
output (Network{..}) act input = V.foldl' f (join [input, 1]) matrices
  where f !inp m = mapVector act $ m <> inp
{-# INLINE output #-}

-- | Computes and keeps the output of all the layers of the neural network with the given activation function
outputs :: (Floating (Vector a), Product a, Storable a, Num (Vector a)) => Network a -> ActivationFunction a -> Vector a -> V.Vector (Vector a)
outputs (Network{..}) act input = V.scanl f (join [input, 1]) matrices
  where f !inp m = mapVector act $ m <> inp
{-# INLINE outputs #-}

deltas :: (Floating (Vector a), Floating a, Product a, Storable a, Num (Vector a)) => Network a -> ActivationFunctionDerivative a -> V.Vector (Vector a) -> Vector a -> V.Vector (Matrix a)
deltas (Network{..}) act' os expected = V.zipWith outer (V.tail ds) (V.init os)
  where !dl = (V.last os - expected) * (deriv $ V.last os)
        !ds = V.scanr f dl (V.zip os matrices)
        f (!o, m) !del = deriv o * (trans m <> del)
        deriv = mapVector act'
{-# INLINE deltas #-}

updateNetwork :: (Floating (Vector a), Floating a, Product a, Storable a, Num (Vector a), Container Vector a) => a -> ActivationFunction a -> ActivationFunctionDerivative a -> Network a -> Sample a -> Network a
updateNetwork alpha act act' n@(Network{..}) (input, expectedOutput) = Network $ V.zipWith (+) matrices corr
    where !xs = outputs n act input
          !ds = deltas n act' xs expectedOutput
          !corr = V.map (scale (-alpha)) ds
{-# INLINE updateNetwork #-}
          
-- | Input vector and expected output vector
type Sample a = (Vector a, Vector a)

-- | List of 'Sample's
type Samples a = [Sample a]

backpropOnce :: (Floating (Vector a), Floating a, Product a, Num (Vector a), Container Vector a) => a -> ActivationFunction a -> ActivationFunctionDerivative a -> Network a -> Samples a -> Network a
backpropOnce rate act act' n samples = foldl' (updateNetwork rate act act') n samples
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
trainUntil :: (Floating (Vector a), Floating a, Product a, Num (Vector a), Container Vector a) => (Int -> Network a -> Samples a -> Bool) -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> Network a -> Samples a -> Network a
trainUntil pr learningRate act act' net samples = go net 0
  where go n !k | pr k n samples = n
                | otherwise      = case backpropOnce learningRate act act' n samples of
                                    n' -> go n' (k+1)
{-# INLINE trainUntil #-}

-- | Trains the neural network with backpropagation the number of times specified by the 'Int' argument,
-- using the given learning rate (second argument).                                   
trainNTimes :: (Floating (Vector a), Floating a, Product a, Num (Vector a), Container Vector a) => Int -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> Network a -> Samples a -> Network a
trainNTimes n = trainUntil (\k _ _ -> k > n)
{-# INLINE trainNTimes #-}

-- | Quadratic error on the given training set using the given activation function. Useful to create
-- your own predicates for 'trainUntil'.
quadError :: (Floating (Vector a), Floating a, Num (Vector a), Num (RealOf a), Product a) => ActivationFunction a -> Network a -> Samples a -> RealOf a
quadError act net samples = foldl' (\err (inp, out) -> err + (norm2 $ output net act inp - out)) 0 samples
{-# INLINE quadError #-}

-- | Trains the neural network until the quadratic error ('quadError') comes below the given value (first argument),
-- using the given learning rate (second argument).
-- 
-- /Note/: this can loop pretty much forever when you're using a bad architecture for the problem, or unappropriate activation
-- functions.
trainUntilErrorBelow :: (Floating (Vector a), Floating a, Product a, Num (Vector a), Ord a, Container Vector a, Num (RealOf a), a ~ RealOf a, Show a) => a -> a -> ActivationFunction a -> ActivationFunctionDerivative a -> Network a -> Samples a -> Network a
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

-- | Derivative of the 'tanh' function from the Prelude.
tanh' :: Floating a => a -> a
tanh' !x = case tanh x of
  s -> 1 - s**2
{-# INLINE tanh' #-}

-- | Loading a neural network from a file (uses zlib compression on top of serialization using the binary package).
--   Will throw an exception if the file isn't there.
loadNetwork :: (Storable a, Element a, Binary a) => FilePath -> IO (Network a)
loadNetwork fp = return . decode . decompress =<< B.readFile fp
{-# INLINE loadNetwork #-}

-- | Saving a neural network to a file (uses zlib compression on top of serialization using the binary package).
saveNetwork :: (Storable a, Element a, Binary a) => FilePath -> Network a -> IO ()
saveNetwork fp net = B.writeFile fp . compress $ encode net
{-# INLINE saveNetwork #-}