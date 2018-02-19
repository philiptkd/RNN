import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Set;

import org.apache.commons.io.FileUtils;

public class Network {	
	//variables to hold numbers of images
	public int numBytes = 0;
	public RNN rnn;
	public char[] fullText;
	public char[] vocab;
	public HashMap<Character, Integer> charToIndex;
	private static final String fileName = "smallFrankenstein.txt";
	
	public Network(boolean loadData) throws IOException {
		//load data
		if(loadData)
			this.loadData();
		
		//write back for testing purposes
		//FileUtils.writeByteArrayToFile(new File("testOut.txt"), this.fullText);
	}
	
	public void trainNet(int epochs, int miniBatchSize, double learningRate) throws IOException {
		for(int epoch=0; epoch<epochs; epoch++) {	//for each epoch
			//for now, it just ignores the last few characters if miniBatchSize*attentionSpan doesn't divide evenly into numChars
			System.out.println("epoch: " + epoch);
			for(int miniBatch=0; miniBatch<this.numBytes/(this.rnn.attentionSpan*miniBatchSize); miniBatch++) { //for each miniBatch
				for(int span=0; span<miniBatchSize; span++) {	//for each attentionSpan of data in this miniBatch
					//put next attentionSpan+1 worth of input data into inputActivations
					loadInputActivations(miniBatch, miniBatchSize, span);
					
					//feed forward
					this.rnn.feedForward();
					
					//backpropagate
					this.rnn.backPropagate();
				}
				//update weights and biases after every miniBatch
				this.rnn.updateWeightsAndBiases(miniBatchSize, learningRate);
				
				//set all the gradients back to zero
				this.resetGradients();
			}
		}
	}
	
	private void resetGradients() {
		this.zeroArray(this.rnn.hiddenBiasGrad);
		this.zeroArray(this.rnn.outputBiasGrad);
		this.zeroArray(this.rnn.wohGrad);
		this.zeroArray(this.rnn.whiGrad);
		this.zeroArray(this.rnn.whhGrad);
	}
	
	private void loadInputActivations(int miniBatch, int miniBatchSize, int span) {
		//zero input activations
		zeroArray(this.rnn.inputActivations);
		
		//set one-hot vectors
		for(int i=0; i<this.rnn.attentionSpan + 1; i++) {
			char thisChar = this.fullText[this.rnn.attentionSpan*(miniBatchSize*miniBatch + span) + i];
			int thisIndex = charToIndex.get(thisChar);
			this.rnn.inputActivations[i][thisIndex] = 1;
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
		
		//print to console
		for(int i=0; i<outputBytes.length; i++) {
			System.out.print((char)outputBytes[i]);
		}
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
        	this.vocab = getVocabulary(fullText);	//gets list of unique characters
        	this.fullText = fullText.toCharArray();
        	this.numBytes = this.fullText.length;
        	
        	for(int i=0; i<this.vocab.length; i++) {	//creates map from character to index in vocab
        		this.charToIndex.put(this.vocab[i], i);
        	}
        }
	}
	
	//returns a char[] of all the unique characters in the input string
	//taken from https://stackoverflow.com/questions/4989091/removing-duplicates-from-a-string-in-java
	private char[] getVocabulary(String string) {
		char[] chars = string.toCharArray();
		Set<Character> charSet = new LinkedHashSet<Character>();
		for (char c : chars) {
		    charSet.add(c);
		}

		StringBuilder sb = new StringBuilder();
		for (Character character : charSet) {
		    sb.append(character);
		}
		return sb.toString().toCharArray();
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
