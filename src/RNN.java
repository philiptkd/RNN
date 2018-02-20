import java.util.Random;

public class RNN {
	//types of activation functions
	private static final int RELU = 1;
	private static final int SIGMOID = 2;
	private static final int TANH = 3;
	private static final int SOFTMAX = 4;
	
	//activation functions used
	private static int outputActFn = SOFTMAX;
	private static int hiddenActFn = TANH;
	
	//input activations for each time step
	public double[][] inputActivations;
	
	//hidden and output activations for each time step
	public double[][] hiddenActivations;
	public double[][] outputActivations;
	public double[] hPrev;
	
	//for backpropagation 
	private double[] dy;
	private double[] dh;
	private double[] dhraw;
	public double[] dhnext;
	
	//biases are constant over each set of time steps
	public double[] hiddenBiases;
	public double[] outputBiases;
	
	//weight matrices are constant over each set of time steps
	public double[][] Whi;	//weights between input and hidden layers
	public double[][] Woh; //weights between hidden and output layers
	public double[][] Whh; //weights between hidden layer and itself in the next time step
	
	//running sums of bias gradients
	public double[] hiddenBiasGrad;
	public double[] outputBiasGrad;
	
	//running sums of weights gradients
	public double[][] whiGrad;
	public double[][] wohGrad;
	public double[][] whhGrad;
	
	//constructor parameters
	public int inputLength;
	public int hiddenLength;
	public int outputLength;
	public int attentionSpan;
	
	//adagrad parameters
	public double[][] mWhi;
	public double[][] mWoh;
	public double[][] mWhh;
	public double[] mbh;
	public double[] mbo;
	
	//random number generator
	private static Random rand = new Random();
	
	//constructor
	public RNN(int inputLength, int hiddenLength, int outputLength, int attentionSpan) throws LayerException {
		//make sure input and output have the same length
		if(inputLength != outputLength) {
			throw new LayerException("inputLength and outputLength must be the same.");
		}
		
		//save constructor parameters
		this.inputLength = inputLength;
		this.hiddenLength = hiddenLength;
		this.outputLength = outputLength;
		this.attentionSpan = attentionSpan;
		
		//create the time-dependent things
		this.inputActivations = new double[attentionSpan+1][inputLength];	//also used as correct answers
		this.hiddenActivations = new double[attentionSpan][hiddenLength];
		this.outputActivations =  new double[attentionSpan][outputLength];
		
		//create the time-independent things
		this.hiddenBiases = new double[hiddenLength];
		this.outputBiases = new double[outputLength];
		this.Whi = new double[hiddenLength][inputLength];
		this.Woh = new double[outputLength][hiddenLength];
		this.Whh = new double[hiddenLength][hiddenLength];
		this.hiddenBiasGrad = new double[hiddenLength];
		this.outputBiasGrad = new double[outputLength];
		this.whiGrad = new double[hiddenLength][outputLength];
		this.wohGrad = new double[outputLength][hiddenLength];
		this.whhGrad = new double[hiddenLength][hiddenLength];
		this.hPrev = new double[hiddenLength];
		this.dy = new double[outputLength];
		this.dh = new double[hiddenLength];
		this.dhraw = new double[hiddenLength];
		this.dhnext = new double[hiddenLength];
		
		//create adagrad things
		this.mbh = new double[hiddenLength];
		this.mbo = new double[outputLength];
		this.mWhh = new double[hiddenLength][hiddenLength];
		this.mWhi = new double[hiddenLength][inputLength];
		this.mWoh = new double[outputLength][hiddenLength];
		
		//initialize biases
		for(int i=0; i<hiddenLength; i++) {
			this.hiddenBiases[i] = 0;//RNN.rand.nextGaussian();
		}
		for(int i=0; i<outputLength; i++) {
			this.outputBiases[i] = 0;//RNN.rand.nextGaussian();
		}
		
		//initialize weights to have acceptable variance
		for(int j=0; j<outputLength; j++) {
			for(int k=0; k<hiddenLength; k++) {
				this.Woh[j][k] = RNN.rand.nextGaussian()*0.01;///Math.sqrt(outputLength*hiddenLength);
			}
		}
		for(int j=0; j<hiddenLength; j++) {
			for(int k=0; k<inputLength; k++) {
				this.Whi[j][k] = RNN.rand.nextGaussian()*0.01;///Math.sqrt(hiddenLength*inputLength);
			}
		}
		for(int j=0; j<hiddenLength; j++) {
			for(int k=0; k<hiddenLength; k++) {
				this.Whh[j][k] = RNN.rand.nextGaussian()*0.01;///Math.sqrt(hiddenLength*hiddenLength);
			}
		}
	}
	
	//calculates and stores all activations
	//assumes inputs for all time steps are already loaded into inputActivations
	public double feedForward(int labelIndex) {
		double loss = 0;
		for(int t=0; t<this.attentionSpan; t++) {	//for each timestep
			
			//calculate hidden activations
			for(int k=0; k<this.hiddenLength; k++) {	//for each node in hidden layer
				double tmpZ = 0;
				
				//contribution from input layer
				for(int i=0; i<this.inputLength; i++) {
					tmpZ += this.inputActivations[t][i]*this.Whi[k][i];
				}
				
				//contribution from previous hidden layer
				if(t > 0) {
					for(int kp=0; kp<this.hiddenLength; kp++) {
						tmpZ += this.hiddenActivations[t-1][kp]*this.Whh[k][kp];
					}
				}
				else {
					for(int kp=0; kp<this.hiddenLength; kp++) {
						tmpZ += this.hPrev[kp]*this.Whh[k][kp];
					}
				}
				
				//add bias
				tmpZ += this.hiddenBiases[k];
				
				//pass through activation function
				this.hiddenActivations[t][k] = hiddenActFn(tmpZ);
			}
		
			//calculate output activations
			for(int j=0; j<this.outputLength; j++) {
				double tmpZ = 0;
				
				//weighted sum from hidden layer
				for(int k=0; k<this.hiddenLength; k++) {
					tmpZ += this.hiddenActivations[t][k]*this.Woh[j][k];
				}
				
				//add bias
				tmpZ += this.outputBiases[j];
				
				//pass through activation function
				this.outputActivations[t][j] = outputActFn(tmpZ);
			}
			if(RNN.outputActFn == RNN.SOFTMAX) {	//normalize softmax
				double sumExp = 0;
				for(int j=0; j<this.outputLength; j++) {
					sumExp += this.outputActivations[t][j];
				}
				for(int j=0; j<this.outputLength; j++) {
					this.outputActivations[t][j] /= sumExp;
				}
			}
			
			loss += -Math.log(this.outputActivations[t][labelIndex]);
		}
		return loss;
	}

	//calculates errors and gradients for each time step
	//uses a mean squared error cost function
	public void backPropagate() {
		for(int t=this.attentionSpan-1; t>=0; t--) { //for each time step, going backwards
			//calculate dy
			for(int j=0; j<this.outputLength; j++) {
				//the input and output have the same length
				//because the data is sequential, the input array also holds the correct answers
				this.dy[j] = this.outputActivations[t][j] - this.inputActivations[t+1][j];
			}			
			
			//calculate Woh gradients
			for(int j=0; j<this.outputLength; j++) {
				for(int k=0; k<this.hiddenLength; k++) {
					this.wohGrad[j][k] += this.dy[j]*this.hiddenActivations[t][k];
				}
			}
			
			//calculate output bias gradients
			for(int j=0; j<this.outputLength; j++) {
				this.outputBiasGrad[j] += this.dy[j];
			}
			
			//calculate dh
			for(int k=0; k<this.hiddenLength; k++) {
				this.dh[k] = 0;
				
				for(int j=0; j<this.outputLength; j++) {
					this.dh[k] += this.dy[j]*this.Woh[j][k] + dhnext[k];
				}
			}
			
			//calculate dhraw. backpropagation through tanh 
			for(int k=0; k<this.hiddenLength; k++) {
				dhraw[k] = (1 - this.hiddenActivations[t][k]*this.hiddenActivations[t][k])*this.dh[k];
			}
			
			//calculate hidden bias gradients
			for(int k=0; k<this.hiddenLength; k++) {
				this.hiddenBiasGrad[k] += this.dhraw[k];
			}
		
			//calculate Whi gradients
			for(int k=0; k<this.hiddenLength; k++) {
				for(int i=0; i<this.inputLength; i++) {
					this.whiGrad[k][i] += this.dhraw[k]*this.inputActivations[t][i];
				}
			}
			
			//calculate Whh gradients
			if(t > 0) {
				for(int kp=0; kp<this.hiddenLength; kp++) {
					for(int k=0; k<this.hiddenLength; k++) {
						this.whhGrad[kp][k] += this.dhraw[kp]*this.hiddenActivations[t-1][k];
					}
				}
			}
			else {
				for(int kp=0; kp<this.hiddenLength; kp++) {
					for(int k=0; k<this.hiddenLength; k++) {
						this.whhGrad[kp][k] += this.dhraw[kp]*this.hPrev[k];
					}
				}
			}
			
			//calculate dhnext		
			// dL/dha[t-1]_k = Sum_i[from outputs i] + Sum_j[from future hiddens j]
			// Sum_j[from future hiddens j] = Sum_j[(dL/dhz[t]_j)(dhz[t]_j/dha[t-1]_k)]
				// = Sum_j[(dL/dhz[t]_j)*Whh_j,k]
			for(int k=0; k<this.hiddenLength; k++) {
				this.dhnext[k] = 0;
				for(int j=0; j<this.hiddenLength; j++) {
					this.dhnext[k] +=  this.dhraw[j]*this.Whh[j][k];
				}
			}
		}
	}

	//prevents gradients from exploding
	public void clipGradients(double extremum) {
		extremum = Math.abs(extremum);
		
		//output biases
		for(int j=0; j<this.outputLength; j++) {
			if(this.outputBiasGrad[j] > extremum) {
				this.outputBiasGrad[j] = extremum;
			}
			else if(this.outputBiasGrad[j] < -extremum) {
				this.outputBiasGrad[j] = -extremum;
			}
		}
		
		//Woh
		for(int j=0; j<this.outputLength; j++) {
			for(int k=0; k<this.hiddenLength; k++) {
				if(this.wohGrad[j][k] > extremum) {
					this.wohGrad[j][k] = extremum;
				}
				else if(this.wohGrad[j][k] < -extremum) {
					this.wohGrad[j][k] = -extremum;
				}
			}
		}
		
		//hidden biases
		for(int k=0; k<this.hiddenLength; k++) {
			if(this.hiddenBiasGrad[k] > extremum) {
				this.hiddenBiasGrad[k] = extremum;
			}
			else if(this.hiddenBiasGrad[k] < -extremum) {
				this.hiddenBiasGrad[k] = -extremum;
			}
		}
		
		//Whh
		for(int j=0; j<this.hiddenLength; j++) {
			for(int k=0; k<this.hiddenLength; k++) {
				if(this.whhGrad[j][k] > extremum) {
					this.whhGrad[j][k] = extremum;
				}
				else if(this.whhGrad[j][k] < -extremum) {
					this.whhGrad[j][k] = -extremum;
				}
			}
		}
		
		//Whi
		for(int j=0; j<this.hiddenLength; j++) {
			for(int k=0; k<this.inputLength; k++) {
				if(this.whiGrad[j][k] > extremum) {
					this.whiGrad[j][k] = extremum;
				}
				else if(this.whiGrad[j][k] < -extremum) {
					this.whiGrad[j][k] = -extremum;
				}
			}
		}
		
	}
	
	//uses adagrad
	public void updateWeightsAndBiases(double learningRate) {
		//output biases
		for(int j=0; j<this.outputLength; j++) {
			this.mbo[j] += this.outputBiasGrad[j]*this.outputBiasGrad[j];
			this.outputBiases[j] += -learningRate*this.outputBiasGrad[j]/Math.sqrt(this.mbo[j] + 1e-8);
		}
		
		//Woh
		for(int j=0; j<this.outputLength; j++) {
			for(int k=0; k<this.hiddenLength; k++) {
				this.mWoh[j][k] += this.wohGrad[j][k]*this.wohGrad[j][k];
				this.Woh[j][k] += -learningRate*this.wohGrad[j][k]/Math.sqrt(this.mWoh[j][k] + 1e-8);
			}
		}
		
		//hidden biases
		for(int k=0; k<this.hiddenLength; k++) {
			this.mbh[k] += this.hiddenBiasGrad[k]*this.hiddenBiasGrad[k];
			this.hiddenBiases[k] += -learningRate*this.hiddenBiasGrad[k]/Math.sqrt(this.mbh[k] + 1e-8);
		}
		
		//Whh
		for(int j=0; j<this.hiddenLength; j++) {
			for(int k=0; k<this.hiddenLength; k++) {
				this.mWhh[j][k] += this.whhGrad[j][k]*this.whhGrad[j][k];
				this.Whh[j][k] += -learningRate*this.whhGrad[j][k]/Math.sqrt(this.mWhh[j][k] + 1e-8);
			}
		}
		
		//Whi
		for(int j=0; j<this.hiddenLength; j++) {
			for(int k=0; k<this.inputLength; k++) {
				this.mWhi[j][k] += this.whiGrad[j][k]*this.whiGrad[j][k];
				this.Whi[j][k] += -learningRate*this.whiGrad[j][k]/Math.sqrt(this.mWhi[j][k] + 1e-8);
			}
		}
	}
	
	private double outputActFn(double z) {
		if(RNN.outputActFn == RNN.RELU) {
			return Math.max(0, z);		//ReLU
		}
		else if(RNN.outputActFn == RNN.SIGMOID) {
			return 1/(1+Math.exp(-z));	//sigmoid
		}	
		else if(RNN.outputActFn == RNN.SOFTMAX) {
			return Math.exp(z);	//not the full softmax function. caller should normalize
		}
		else
			return 0;
	}
	
	private double hiddenActFn(double z) {
		if(RNN.hiddenActFn == RNN.RELU) {
			return Math.max(0, z);		//ReLU
		}
		else if(RNN.hiddenActFn == RNN.SIGMOID) {
			return 1/(1+Math.exp(-z));	//sigmoid
		}
		else if(RNN.hiddenActFn == RNN.TANH) {
			return Math.tanh(z);
		}
		else
			return 0;
	}
	
	private double actFnPrime(double a, int actFn) {
		if(actFn == RNN.RELU) {
			if(a > 0) {
				return 1;
			}
			else
				return 0;
		}
		else if(actFn == RNN.SIGMOID) {
			return a*(1-a);
		}	
		else
			return 0;
	}
}
