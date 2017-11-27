import java.util.Random;

public class RNN {
	//types of activation functions
	private static final int RELU = 1;
	private static final int SIGMOID = 2;
	
	//activation functions used
	private static int outputActFn = SIGMOID;
	private static int hiddenActFn = RELU;
	
	//input activations for each time step
	private double[][] inputActivations;
	
	//hidden and output deltas and activations for each time step
	private double[][] hiddenDeltas;
	private double[][] hiddenActivations;
	private double[][] outputDeltas;
	private double[][] outputActivations;
	
	//biases are constant over each set of time steps
	private double[] hiddenBiases;
	private double[] outputBiases;
	
	//weight matrices are constant over each set of time steps
	private double[][] Whi;	//weights between input and hidden layers
	private double[][] Woh; //weights between hidden and output layers
	private double[][] Whh; //weights between hidden layer and itself in the next time step
	
	//running sums of bias gradients
	private double[] hiddenBiasGrad;
	private double[] outputBiasGrad;
	
	//running sums of weights gradients
	private double[][] whiGrad;
	private double[][] wohGrad;
	private double[][] whhGrad;
	
	//constructor parameters
	private int inputLength;
	private int hiddenLength;
	private int outputLength;
	private int attentionSpan;
	
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
		this.hiddenDeltas = new double[attentionSpan][hiddenLength];
		this.hiddenActivations = new double[attentionSpan][hiddenLength];
		this.outputDeltas = new double[attentionSpan][outputLength];
		this.outputActivations =  new double[attentionSpan][outputLength];
		
		//create the time-independent things
		this.hiddenBiases = new double[hiddenLength];
		this.outputBiases = new double[outputLength];
		this.Whi = new double[hiddenLength][outputLength];
		this.Woh = new double[outputLength][hiddenLength];
		this.Whh = new double[hiddenLength][hiddenLength];
		this.hiddenBiasGrad = new double[hiddenLength];
		this.outputBiasGrad = new double[outputLength];
		this.whiGrad = new double[hiddenLength][outputLength];
		this.wohGrad = new double[outputLength][hiddenLength];
		this.whhGrad = new double[hiddenLength][hiddenLength];
		
		//initialize biases
		for(int i=0; i<hiddenLength; i++) {
			this.hiddenBiases[i] = RNN.rand.nextGaussian();
		}
		for(int i=0; i<outputLength; i++) {
			this.outputBiases[i] = RNN.rand.nextGaussian();
		}
		
		//initialize weights to have acceptable variance
		for(int j=0; j<outputLength; j++) {
			for(int k=0; k<hiddenLength; k++) {
				this.Woh[j][k] = RNN.rand.nextGaussian()/Math.sqrt(outputLength*hiddenLength);
			}
		}
		for(int j=0; j<hiddenLength; j++) {
			for(int k=0; k<inputLength; k++) {
				this.Whi[j][k] = RNN.rand.nextGaussian()/Math.sqrt(hiddenLength*inputLength);
			}
		}
		for(int j=0; j<hiddenLength; j++) {
			for(int k=0; k<hiddenLength; k++) {
				this.Whh[j][k] = RNN.rand.nextGaussian()/Math.sqrt(hiddenLength*hiddenLength);
			}
		}
	}
	
	//calculates and stores all activations
	//assumes input is already loaded into inputActivations[timeStep]
	//assumes hiddenActivations[timeStep-1] were already calculated
	public void feedForward(int timeStep) {
		//calculate hidden activations
		for(int k=0; k<this.hiddenLength; k++) {	//for each node in hidden layer
			double tmpZ = 0;
			
			//contribution from input layer
			for(int i=0; i<this.inputLength; i++) {
				tmpZ += this.inputActivations[timeStep][i]*this.Whi[k][i];
			}
			
			//contribution from previous hidden layer
			if(timeStep > 0) {
				for(int kp=0; kp<this.hiddenLength; kp++) {
					tmpZ += this.hiddenActivations[timeStep-1][kp]*this.Whh[k][kp];
				}
			}
			
			//add bias
			tmpZ += this.hiddenBiases[k];
			
			//pass through activation function
			this.hiddenActivations[timeStep][k] = hiddenActFn(tmpZ);
		}
		
		//calculate output activations
		for(int j=0; j<this.outputLength; j++) {
			double tmpZ = 0;
			
			//weighted sum from hidden layer
			for(int k=0; k<this.hiddenLength; k++) {
				tmpZ += this.hiddenActivations[timeStep][k]*this.Woh[j][k];
			}
			
			//add bias
			tmpZ += this.outputBiases[j];
			
			//pass through activation function
			this.outputActivations[timeStep][j] = outputActFn(tmpZ);
		}
	}
	
	//calculates errors and gradients for each time step
	public void backPropagate() {
		for(int t=0; t<this.attentionSpan; t++) { //for each time step
			//calculate output deltas
			for(int j=0; j<this.outputLength; j++) {
				//the input and output have the same length
				//because the data is sequential, the input array also holds the correct answers
				this.outputDeltas[t][j] = this.outputActivations[t][j] - this.inputActivations[t+1][j];
				//multiply by derivative of activation function
				this.outputDeltas[t][j] *= actFnPrime(this.outputActivations[t][j], RNN.outputActFn);
			}
			
			
			//calculate hidden deltas
			for(int k=0; k<this.hiddenLength; k++) {
				//TODO: start here
			}
		}
			
		
		for(int t=0; t<this.attentionSpan; t++) { //for each time step
			//calculate output bias gradients
			
			//calculate hidden bias gradients
		
			//calculate Woh gradients
		
			//calculate Whi gradients
		
			//calculate Whh gradients
		}
	}
	
	private double outputActFn(double z) {
		if(RNN.outputActFn == RNN.RELU) {
			return Math.max(0, z);		//ReLU
		}
		else if(RNN.outputActFn == RNN.SIGMOID) {
			return 1/(1+Math.exp(-z));	//sigmoid
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
