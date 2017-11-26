//for now, holds two FCLayers
//meant to be a stackable network structure for creating deep RNNs
public class RNN {
	private FCLayer inputFC;
	private FCLayer outputFC;
	public RNN(int inputLength, int hiddenLength, int outputLength) {
		//create each set of nodes
		inputFC = new FCLayer(inputLength, hiddenLength);
		outputFC = new FCLayer(hiddenLength, outputLength);
		
		//connect them
		inputFC.next = outputFC;
		outputFC.prev = inputFC;
	}
	
	public void feedForward() {
		if(this.inputFC.prev != null) {
			System.out.println("Should not be calling RNN feedForward from intermediate RNN layer.");
		}
		else {
			this.inputFC.feedForward();
		}
	}
	
	public void backPropagate() {
		if(this.outputFC.next != null) {
			System.out.println("Should not be calling RNN backPropagate from intermediate RNN layer.");
		}
		else {
			this.outputFC.backpropagate();
		}
	}
}
