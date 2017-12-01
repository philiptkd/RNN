import java.io.IOException;

//uses a simple RNN to predict the next character in a string

public class Main {

	public static void main(String[] args) {
		try {
			RNN rnn = new RNN(128, 3*128, 128, 20);
			Network net = new Network(rnn);
			net.trainNet(1, 1, 3.0);
			net.generate('A', 500);
		} catch (LayerException e) {
			System.out.println(e.getMessage());
		} catch (IOException e) {
			System.out.println(e.getMessage());
		}
		
		
		
	}

}
