package milestone2;

// Reading from .arff data file
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.classifiers.functions.*;
import weka.classifiers.functions.supportVector.*;

public class SMO {

    public static void main(String[] args) throws IOException {

        BufferedReader reader = new BufferedReader(new FileReader("data/anneal_train.arff"));
        Instances data = new Instances(reader);
        reader.close();
        // setting class attribute
        data.setClassIndex(data.numAttributes() - 1);

        // TODO Auto-generated method stub
        SMO smo = new SMO();
        // set further options via set-methods
        PolyKernel poly  =new PolyKernel();
        // set further options via set-methods
        //smo.setKernel(poly);
        System.out.println(poly);
        System.out.println(smo);
        System.out.print(data);
    }

}
