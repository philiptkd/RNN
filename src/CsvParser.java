import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class CsvParser {
    public static double[][] parse(String fileName, int rows, int cols) {
        double[][] ret = new double[rows][cols];
    	File file= new File(fileName);

        // this gives you a 2-dimensional array of strings
        Scanner inputStream;

        int row = 0;
        try{
            inputStream = new Scanner(file);

            while(inputStream.hasNext()){
                String line= inputStream.next();
                String[] values = line.split(",");
                
                for(int i=0; i<cols; i++) {
                	ret[row][i] = Double.parseDouble(values[i]);
                }
                row++;
            }

            inputStream.close();
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        
        return ret;
    }

}