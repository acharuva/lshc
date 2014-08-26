package org.mlbio.classifier.test;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;
import org.junit.Test;
import org.mlbio.classifier.BinarySVM;
import org.mlbio.classifier.Example;
import org.mlbio.classifier.WeightParameter;



public class BinarySVMTester {
	
	@Test
	public void compareBinarySVM_clef() throws IOException{
		//Compares my binary SVM modification with G.Siddarth's
		String trainingFile = "/Users/acharuva/Projects/lsmtl/data/clef/train.txt";
		Vector<Example> data = readData(trainingFile);
		Vector<gc.base.Example> data2 = readData2(trainingFile);
		

		
		for(int k=0; k<2; k++){
			BinarySVM svm = new BinarySVM();
			WeightParameter weights = WeightParameter.parseString("53 1 0 51", 80);
			svm.optimize(data, weights, 1, 0.01, 10000);
			System.out.println(weights);	
			System.out.println(weights.norm());			
			
			gc.ml.BinarySVM svm2 = new gc.ml.BinarySVM();
			gc.base.WeightParameter weights2  = new gc.base.WeightParameter(53,81);
			svm2.optimize(data2, weights2, 1, 0.01, 10000);
			System.out.println(weights2);
			System.out.println(weights2.norm());
			
			double normDiff = 0;
			for(int i=0;i<weights.weightvector.length;i++){
				double diff = (weights.weightvector[i] - weights2.weightvector[i]);
				normDiff = diff*diff;
			}
			System.out.println(normDiff);
		}
		
		
	}

    public void testBinarySVM_clef() throws IOException {
        /* inputs: 
         * training file
         * test file
         * positive class label
         * num_dim
         * 
         * return 
         * 
         * */
        String PROJECT_HOME   = "/Users/acharuva/Projects/lsmtl/data/lshtc-2010-dmoz-small/";
        String trainingFile   = PROJECT_HOME + "xtrain/xtrain.txt";
        String testFile       = PROJECT_HOME + "/validation/validation.txt";
        int positiveLabel     = 49552;
        float C               = 10;
        int numDimensions     = 51033;
        boolean useTrainForTest = true;
        
        
//        String dataDir = "/Users/acharuva/Projects/lsmtl/data/lshtc-2010-dmoz-small/";
//        String trainingFile = dataDir + "xtrain/xtrain.txt";
//        String testFile = dataDir + "/validation/validation.txt";
//        int positiveLabel = 33;
//        int numDimensions = 51033;
//        float C = 10;
        
        
        Vector<Example> trainingData = readData(trainingFile);
        Vector<Example> testData = null;
        if(useTrainForTest){
            testData = trainingData;
        }else{
            testData = readData(testFile);    
        }
        
        WeightParameter param = WeightParameter.parseString(
                    (""+positiveLabel + " 1 0 2 2"), 
                    numDimensions+1);
        BinarySVM svm = new BinarySVM();
        System.out.println("Training SVM");
        svm.optimize(trainingData, param, C, 0.001 , 1000);
        
        int TP = 0;
        int FP = 0;
        int TN = 0;
        int FN = 0;
        for(Example e:testData){
            double score = param.getScore(e);
            if(e.labels[0] == param.node){
                if(score > 0)
                    TP += 1;
                else
                    FN += 1;
            }else{
                if(score < 0)
                    TN += 1;
                else
                    FP += 1;
            }
            
        }
        System.out.println("Confusion Matrix");
        System.out.println(""+TP+"\t"+FN + "\n"+FP + "\t"+TN);
    }
    
    
    public static Vector<gc.base.Example> readData2(String inputFile) throws IOException{
        //read example data
        Vector<gc.base.Example> data = new Vector<gc.base.Example>();        
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line;
        while((line=reader.readLine()) != null){
            line = line.replaceAll(System.getProperty("line.separator"), "");
            gc.base.Example e = new gc.base.Example(line);
//            e.normalize();
            data.add(e);
        }        
        reader.close();
        return data;
    }

    public static Vector<Example> readData(String inputFile) throws IOException{
        //read example data
        Vector<Example> data = new Vector<Example>();        
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line;
        while((line=reader.readLine()) != null){
            line = line.replaceAll(System.getProperty("line.separator"), "");
            Example e = new Example(line);
//            e.normalize();
            data.add(e);
        }        
        reader.close();
        return data;
    }
    
    public static void testSVMTime(String[] args) throws IOException{
    	//test the time required for running a single SVM on datasets
    	
    	String trainingFile = args[0];
    	int node = Integer.parseInt(args[1]);
    	int numDim = Integer.parseInt(args[2]);
    	
//    	String trainingFile = "/Users/acharuva/Projects/lsmtl/data/clef/train.txt";
//    	int node = 53;
//    	int numDim = 80;
    	
    	long startTime = System.nanoTime();
		Vector<Example> data = readData(trainingFile);
		long endTime = System.nanoTime();
		double dsReadTime = (endTime-startTime)/Math.pow(10,9);
		
		startTime = System.nanoTime();
		BinarySVM svm = new BinarySVM();
		WeightParameter weights = new WeightParameter( node, numDim+1);
		svm.optimize(data, weights, 1, 0.01, 1000);
		endTime = System.nanoTime();
		double trainTime = (endTime-startTime)/Math.pow(10,9);
		System.out.println(
				trainingFile + 
				" & Load DS time = "+dsReadTime + 
				" & Train time = " + trainTime  
				);
    }
   
    

    public static void main(String[] args) throws IOException {
		testSVMTime(args);
	}
    
}
