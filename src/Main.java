import java.io.IOException;

//uses a simple RNN to predict the next character in a string

public class Main {

	public static void main(String[] args) {
		try {
			Network net = new Network(true);
			RNN rnn = new RNN(net.vocab.length, 100, net.vocab.length, 25);	//inputLength, hiddenLength, outputLength, attentionSpan
			net.rnn = rnn;
			
			int position = 0;
			int n = 0;
			double smooth_loss = -Math.log(1.0/net.vocab.length)*rnn.attentionSpan;
			
			while(true) {
				if(position + rnn.attentionSpan + 1 >= net.fullText.length) {
					net.zeroArray(rnn.hPrev);	//reset hPrev
					position = 0;
				}
				
				if(n%1000 == 0) 
					net.sample(net.fullText[position], 200);		//seed sample with next character so we can make use of hPrev
				
				double loss = net.trainNet(position, 0.1);	//position, learningRate
				smooth_loss = smooth_loss*0.999 + loss*0.001;
				if(n%1000 == 0)
					System.out.println(smooth_loss+"\n");
				
				position += rnn.attentionSpan;
				n++;
			}
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
