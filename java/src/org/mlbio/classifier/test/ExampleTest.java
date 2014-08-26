package org.mlbio.classifier.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Vector;

import org.junit.Test;
import org.mlbio.classifier.Example;

public class ExampleTest {

    @Test
    public void readExamples() throws IOException {
        
        //read example data
        Vector<Example> data = new Vector<Example>();
        String inputFile = "/Users/acharuva/Projects/lsmtl/data/tmp_test_clef/clef.test.normalized.svm";
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        String line;
        while((line=reader.readLine()) != null){
            line = line.replaceAll(System.getProperty("line.separator"), "");
            Example e = new Example(line);
            data.add(e);
            System.out.println(e.toString());
        }
                
    }    

}
