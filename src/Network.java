import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import org.apache.commons.io.FileUtils;

public class Network {	
	//variables to hold numbers of images
	private int numBytes = 0;
	private RNN rnn;
	private byte[] fullText;
	private static final String fileName = "smallFrankenstein.txt";
	
	public Network(RNN rnn) throws IOException {
		//save input RNN
		this.rnn = rnn;
		
		//load data
		this.loadData();
		
		//write back for testing purposes
		//FileUtils.writeByteArrayToFile(new File("testOut.txt"), this.fullText);
	}
	
	public void trainNet(int epochs, int miniBatchSize, double learningRate) throws IOException {
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			//for now, it just ignores the last few characters if miniBatchSize*attentionSpan doesn't divide evenly into numChars
			for(int miniBatch=0; miniBatch<this.numBytes/(this.rnn.attentionSpan*miniBatchSize); miniBatch++) { //for each miniBatch
				//put next attentionSpan+1 worth of input data into inputActivations
				loadInputActivations(miniBatch, miniBatchSize);
				
				//feed forward
				this.rnn.feedForward();
				
				//backpropagate
				this.rnn.backPropagate();
			}
			
			//update weights and biases
			this.rnn.updateWeightsAndBiases(miniBatchSize, learningRate);
			
			//set all the gradients back to zero
			this.resetGradients();
		}
	}
	
	private void resetGradients() {
		this.zeroArray(this.rnn.hiddenBiasGrad);
		this.zeroArray(this.rnn.outputBiasGrad);
		this.zeroArray(this.rnn.wohGrad);
		this.zeroArray(this.rnn.whiGrad);
		this.zeroArray(this.rnn.whhGrad);
	}
	
	private void loadInputActivations(int miniBatch, int miniBatchSize) {
		//zero input activations
		zeroArray(this.rnn.inputActivations);
		
		//set one-hot vectors
		for(int i=0; i<this.rnn.attentionSpan + 1; i++) {
			byte thisChar = this.fullText[this.rnn.attentionSpan*miniBatchSize + i];
			this.rnn.inputActivations[i][thisChar] = 1;
		}
	}
	
	private void zeroArray(double[][] arr) {
		for(int i=0; i<arr.length; i++) {
			for(int j=0; j<arr[0].length; j++) {
				arr[i][j] = 0;
			}
		}
	}
	
	private void zeroArray(double[] arr) {
		for(int i=0; i<arr.length; i++) {
			arr[i] = 0;
		}
	}

	public void generate(char seed, int len) throws IOException {
		//create output byte array
		byte[] outputBytes = new byte[len];
		
		//load seed
		this.zeroArray(this.rnn.inputActivations[0]);
		this.rnn.inputActivations[0][(byte)seed] = 1;
		
		//generate len bytes
		for(int i=0; i<len; i++) {
			int layer;
			if(i == 0)
				layer = 0;
			else
				layer = 1;
			
			//feed forward once
			this.rnn.feedForwardOnce(layer);
			
			//get output byte and put in array
			byte classification = 0;
			double highest = Double.MIN_VALUE;
			for(int j=0; j<this.rnn.outputLength; j++) {
				if(this.rnn.outputActivations[layer][j] > highest) {
					highest = this.rnn.outputActivations[layer][j];
					classification = (byte)j;
				}
			}
			outputBytes[i] = classification;
			
			//load one-hot version of output as next input
			this.zeroArray(this.rnn.inputActivations[1]);
			this.rnn.inputActivations[1][classification] = 1;
			
			//move hidden activations
			if(layer == 1) {
				for(int k=0; k<this.rnn.hiddenLength; k++) {
					this.rnn.hiddenActivations[0][k] = this.rnn.hiddenActivations[1][k];
				}
			}
		}
		//write byte array
		FileUtils.writeByteArrayToFile(new File("testOut.txt"), outputBytes);
	}
	
	//grabbed from https://www.caveofprogramming.com/java/java-file-reading-and-writing-files-in-java.html#readtext
	//puts text file into byte array
	private void loadData() {
		 // This will reference one line at a time
        String line = "";
        String fullText = "";

        try {
            // FileReader reads text files in the default encoding.
            FileReader fileReader = 
                new FileReader(fileName);

            // Always wrap FileReader in BufferedReader.
            BufferedReader bufferedReader = 
                new BufferedReader(fileReader);

            while((line = bufferedReader.readLine()) != null) {
                fullText += System.getProperty("line.separator") + line;
            }   

            // Always close files.
            bufferedReader.close();         
        }
        catch(FileNotFoundException ex) {
            System.out.println(
                "Unable to open file '" + 
                fileName + "'");                
        }
        catch(IOException ex) {
            System.out.println(
                "Error reading file '" 
                + fileName + "'");                  
        }
        
        if(fullText != null) {
        	this.fullText = fullText.getBytes(StandardCharsets.US_ASCII); // Java 7+ only
        	this.numBytes = this.fullText.length;
        }
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
