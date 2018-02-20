import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.Random;
import java.util.Set;

import org.apache.commons.io.FileUtils;

public class Network {	
	//variables to hold numbers of images
	public int numBytes = 0;
	public RNN rnn;
	public char[] fullText;
	public char[] vocab;
	public HashMap<Character, Integer> charToIndex = new HashMap<Character, Integer>();
	private static final String fileName = "smallFrankenstein.txt";
	private static Random rand = new Random();
	
	public Network(boolean loadData) throws IOException {
		//load data
		if(loadData)
			this.loadData();
		
		//write back for testing purposes
		//FileUtils.writeByteArrayToFile(new File("testOut.txt"), this.fullText);
	}
	
	public double trainNet(int position, double learningRate) throws IOException {
		//load input
		loadInputActivations(position);
		
		//feed forward
		double loss = this.rnn.feedForward(this.charToIndex.get(this.fullText[position+1]));
		
		//set all the gradients back to zero
		this.resetGradients();
		
		//reset dhnext
		this.zeroArray(this.rnn.dhnext);
		
		//backpropagate
		this.rnn.backPropagate();
		
		//save hPrev
		for(int k=0; k<this.rnn.hiddenLength; k++) {
			this.rnn.hPrev[k] = this.rnn.hiddenActivations[this.rnn.attentionSpan-1][k];
		}
	
		//clip gradients
		this.rnn.clipGradients(5);
		
		//update weights and biases
		this.rnn.updateWeightsAndBiases(learningRate);
		
		return loss;
	}
	
	private void resetGradients() {
		this.zeroArray(this.rnn.hiddenBiasGrad);
		this.zeroArray(this.rnn.outputBiasGrad);
		this.zeroArray(this.rnn.wohGrad);
		this.zeroArray(this.rnn.whiGrad);
		this.zeroArray(this.rnn.whhGrad);
	}
	
	private void loadInputActivations(int position) {
		//zero input activations
		zeroArray(this.rnn.inputActivations);
		
		//set one-hot vectors
		for(int i=0; i<this.rnn.attentionSpan + 1; i++) {
			char thisChar = this.fullText[position + i];
			int thisIndex = charToIndex.get(thisChar);
			this.rnn.inputActivations[i][thisIndex] = 1;
		}
	}
	
	public void zeroArray(double[][] arr) {
		for(int i=0; i<arr.length; i++) {
			for(int j=0; j<arr[0].length; j++) {
				arr[i][j] = 0;
			}
		}
	}
	
	public void zeroArray(double[] arr) {
		for(int i=0; i<arr.length; i++) {
			arr[i] = 0;
		}
	}

	public void sample(char seed, int len) {
		double[] x = new double[this.rnn.inputLength];
		double[] hPrev = new double[this.rnn.hiddenLength];
		
		//first input
		int index = this.charToIndex.get(seed);
		x[index] = 1;
		
		//initialize hPrev
		for(int k=0; k<this.rnn.hiddenLength; k++) {
			hPrev[k] = this.rnn.hPrev[k];
		}
		
		for(int t=0; t<len; t++) {
			double[] h = new double[this.rnn.hiddenLength];
			double[] y = new double[this.rnn.outputLength];
			
			//feed forward
			for(int k=0; k<this.rnn.hiddenLength; k++) {	//hidden activations
				for(int i=0; i<this.rnn.inputLength; i++) {
					h[k] += this.rnn.Whi[k][i]*x[i];
				}
				for(int i=0; i<this.rnn.hiddenLength; i++) {
					h[k] += this.rnn.Whh[k][i]*hPrev[i];
				}
				h[k] += this.rnn.hiddenBiases[k];
				h[k] = Math.tanh(h[k]);
			}
			double sumExp = 0;
			for(int j=0; j<this.rnn.outputLength; j++) {	//output activations
				for(int k=0; k<this.rnn.hiddenLength; k++) {
					y[j] += this.rnn.Woh[j][k]*h[k];
				}
				y[j] += this.rnn.outputBiases[j];
				y[j] = Math.exp(y[j]);
				sumExp += y[j];
			}
			for(int j=0; j<this.rnn.outputLength; j++) {
				y[j] /= sumExp;
			}
			
			//randomly sample 
			double sample = Network.rand.nextDouble();
			double sum = 0;
			for(int j=0; j<this.rnn.outputLength; j++) {
				sum += y[j];
				if(sum > sample) {
					index = j;
					break;
				}
			}
			
			//print and set new input
			System.out.print(this.vocab[index]);
			this.zeroArray(x);
			x[index] = 1;
			
			//set new hPrev
			for(int k=0; k<this.rnn.hiddenLength; k++) {
				hPrev[k] = h[k];
			}
		}
		System.out.println("");
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
