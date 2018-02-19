import java.io.IOException;

//uses a simple RNN to predict the next character in a string

public class Main {

	public static void main(String[] args) {
		try {
			Network net = new Network(true);
			RNN rnn = new RNN(net.vocab.length, 100, net.vocab.length, 25);	//inputLength, hiddenLength, outputLength, attentionSpan
			net.rnn = rnn;
			
			//RNN rnn = new RNN(2,3,2,2);
			//setToyWeights(rnn);
			//Network net = new Network(rnn, false);
			//loadToyInput(net);
			
			net.trainNet(10, 1, 0.1);	//epochs, miniBatchSize, learningRate
			net.generate('A', 500);		//seed, len
		} catch (LayerException e) {
			System.out.println(e.getMessage());
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
	}
	
	public static void loadToyInput(Network net) {
		char[] fullText = {1,0,1,0,1};
		net.fullText = fullText;
		net.numBytes = fullText.length;
	}
	
	public static void setToyWeights(RNN rnn) {
		double[][] whi = {{-1,-.4},{-.5,-.9},{.6,.4}};
		rnn.Whi = whi;
		
		double[][] whh = {{-.4,-.6,-.9},{.2,.9,-.3},{-.2,-.3,.9}};
		rnn.Whh = whh;
		
		double[][] woh = {{.9,.4,-.9},{-.3,.1,.5}};
		rnn.Woh = woh;
		
		double[] hiddenBiases = {0,0,0};
		rnn.hiddenBiases = hiddenBiases;
		
		double[] outputBiases = {0,0};
		rnn.outputBiases = outputBiases;
	}

}
