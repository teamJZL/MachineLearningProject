package milestone3;

import weka.core.*;
import weka.classifiers.*;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.*;
import weka.classifiers.trees.*;

import java.io.*;

/**
 * With modifications from http://weka.wikispaces.com/Optimizing+parameters
 * 
 * A little example for optimizing J48's confidence parameter with 
 * CVPArameterSelection meta-classifier.
 * The class expects a dataset as first parameter, class attribute is
 * assumed to be the last attribute.
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class CVParam {
   public static void main(String[] args) throws Exception {
      // load data
      BufferedReader reader = new BufferedReader(new FileReader("data/anneal_train.arff")); //changed from args[0] to datafile
      Instances data = new Instances(reader);
      reader.close();
      data.setClassIndex(data.numAttributes() - 1);

      // setup classifier
      CVParameterSelection ps = new CVParameterSelection();
      //ps.setClassifier(new J48());
      //ps.setNumFolds(5);  // using 5-fold CV
      //ps.addCVParameter("C 0.1 0.5 5");
      
      ps.setClassifier(new SMO());
      ps.setNumFolds(10);
      //ps.addCVParameter("C 2 8 4"); // 2,4,6,8 = 4 steps
      ps.addCVParameter("C 2 16 8"); // 2,4,6,8 = 4 steps
      //Scheme:weka.classifiers.functions.SMO -C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0"
      
      // build and output best options
      ps.buildClassifier(data);
      System.out.println(Utils.joinOptions(ps.getBestClassifierOptions()));
   }
}