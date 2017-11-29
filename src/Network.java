import java.io.IOException;
import java.io.RandomAccessFile;

//TODO
//in trainNet, put input activations into first layer of first RNN

public class Network {
	//////////////comments are from Network class in ConvNet project////////////////
	
	//variables to hold numbers of images
	private int numChars;
	private RNN rnn;
	
	//arrays to hold images and labels
	//strings of file names
	//pointers to first and last layers
	//static random number generator
	
	public Network() {
		//save list of layers
		//save first and last layers separately for ease of use
		//save file name strings
		//try to connect all the layers in the ordered list of layers
		//count the amount of input data and load it in
	}
	
	public void trainNet(int epochs, int miniBatchSize, double learningRate) throws IOException {
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			//for now, it just ignores the last few characters if miniBatchSize*attentionSpan doesn't divide evenly into numChars
			for(int miniBatch=0; miniBatch<this.numChars/(this.rnn.attentionSpan*miniBatchSize); miniBatch++) { //for each miniBatch
				//put next attentionSpan+1 worth of input data into inputActivations
				loadInputActivations(miniBatch);
				
				//feed forward
				this.rnn.feedForward();
				
				//backpropagate
				this.rnn.backPropagate();
			}
			//update weights and biases
			this.rnn.updateWeightsAndBiases();
		}
	}
	
	private void loadInputActivations(int miniBatch) {
		
	}
	
	public void testNet(int dataSet) throws IOException {
		
	}
		
	public void getNumLines() {
		
	}
	
	private void loadData() {
		
	}
	
	public void writeToFile() {
		
	}
	
	public boolean readFromFile() {
		boolean successfulRead = false;
		
		return successfulRead;
	}
	
	private void stopReading(RandomAccessFile reader) {
		System.out.println("This weights file is incompatible with the current network.");
		try {
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
