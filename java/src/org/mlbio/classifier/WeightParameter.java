package org.mlbio.classifier;

import java.io.*;
import java.util.StringTokenizer;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public class WeightParameter implements Writable {
    public int node;
    public float[] weightvector;
    public int[] neighbors;
    public boolean isLeaf;
    public int trainSetNum = 0; // used for level by level training
    
    // ONLY for DEBGUGGING
    // training objective calculated with
    // whatever dataset is used for training
    public double trainObjective = 0;
    // BinarySVM training time for leaves
    // averaging time for non-leaves
    public double trainTime = 0;
    // time to compute objectiveValues
    public double objCalcTime = 0;

    public int[] getNeighbors() {
        return neighbors;
    }

    public void setNeighbors(int[] neighbors) {
        this.neighbors = neighbors;
    }

    public WeightParameter() {
        clear();
    }
    
    public WeightParameter(int node, int size){
    	this.node = node;
    	weightvector = new float[size];
    }
    

    public WeightParameter(final WeightParameter x) {
        node = x.node;
        weightvector = x.weightvector.clone();
        isLeaf = x.isLeaf;
        neighbors = x.neighbors.clone();
        trainSetNum = x.trainSetNum;
        
    }

    public void clear() {
        node = -1;
        weightvector = new float[1];
        weightvector[0] = 0;
        neighbors = new int[1];
        neighbors[0] = 0;
        trainSetNum = 0;
    }

    public void readFields(DataInput in) throws IOException {
        node = in.readInt();
        int n = in.readInt();

        weightvector = new float[n];
        for (int i = 0; i < n; ++i)
            weightvector[i] = in.readFloat();

        int m = in.readInt();
        neighbors = new int[m];
        for (int i = 0; i < m; i++) {
            neighbors[i] = in.readInt();
        }
        isLeaf = in.readBoolean();
        trainSetNum = in.readInt();
        trainTime = in.readDouble();
        trainObjective = in.readDouble();
        objCalcTime = in.readDouble();        
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(node);
        out.writeInt(weightvector.length);

        for (int i = 0; i < weightvector.length; ++i)
            out.writeFloat(weightvector[i]);

        out.writeInt(neighbors.length);
        for (int i = 0; i < neighbors.length; i++) {
            out.writeInt(neighbors[i]);
        }
        out.writeBoolean(isLeaf);
        out.writeInt(trainSetNum);
        out.writeDouble(trainTime);
        out.writeDouble(trainObjective);
        out.writeDouble(objCalcTime);
    }

    public int getNode() {
        return node;
    }

    public void setNode(int node) {
        this.node = node;
    }

    public float[] getWeightvector() {
        return weightvector;
    }

    public void setWeightvector(float[] weightvector) {
        this.weightvector = weightvector;
    }

    public boolean isLeaf() {
        return isLeaf;
    }

    public void setLeaf(boolean isLeaf) {
        this.isLeaf = isLeaf;
    }

    public double getScore(Example E) {
        double score = 0;
        for (int i = 0; i < E.fids.length; ++i){
        	if(E.fids[i] < weightvector.length){
        		score += weightvector[E.fids[i]] * E.fvals[i];
        	}
        }
                
        return score;
    }

    public void add(WeightParameter p) {
        if (weightvector.length >= p.weightvector.length) {
            for (int i = 0; i < p.weightvector.length; ++i)
                weightvector[i] += p.weightvector[i];
        } else {
            float[] newwv = p.weightvector.clone();
            for (int i = 0; i < weightvector.length; ++i)
                newwv[i] += weightvector[i];
            weightvector = newwv;
        }
    }

    public String toString(int x) {
        String ret = " Node = " + node + "\n wv = " + weightvector.toString()
                + "\n";
        return ret;
    }

    public static WeightParameter parseString(String input, int ndim) {
        // input format
        // node_id is_leaf={0/1} trainSetNum=int [neighbor_node_id]+

        // one additional feature is added for bias
        ndim += 1;  

        
        WeightParameter param = new WeightParameter();
        StringTokenizer tokens = new StringTokenizer(input);

        param.node = Integer.parseInt(tokens.nextToken());
        param.isLeaf = Integer.parseInt(tokens.nextToken()) != 0;
        param.trainSetNum = Integer.parseInt(tokens.nextToken());

        int n = tokens.countTokens();
        param.neighbors = new int[n];
        for (int i = 0; i < n; i++) {
            param.neighbors[i] = Integer.parseInt(tokens.nextToken());
        }

        param.weightvector = new float[ndim];
        for (int i = 0; i < ndim; i++) {
            param.weightvector[i] = 0;
        }
        return param;
    }


    public String toString() {
        String ret = " Node = " + node + " & wv = ";
        for (int i = 0; i < weightvector.length; ++i)
            ret += i + ":" + weightvector[i] + " ";
        return ret;
    }

    public double norm() {
        double ret = 0;
        for (int i = 0; i < weightvector.length; ++i)
            ret += weightvector[i] * weightvector[i];
        return ret;
    }
    
    public double normDiff(WeightParameter other){
        // compute L2 norm of the difference between
        // this - other
        double ret  = 0;
        for(int i=0; i< weightvector.length; i++){
            double diff = (weightvector[i] - other.weightvector[i]);
            ret += diff*diff;
        }
        return ret;
    }
        
    public double hingeLoss(Example e){
        double dotProduct = 0;
        int label = -1;
        for(int l:e.labels){
            if(l==node){
                label = 1;
                break;
            }
        }
        for(int i=0;i<e.fids.length;i++){
            dotProduct += e.fvals[i]*weightvector[e.fids[i]];
        }
        double loss = Math.max(0, 1-label*dotProduct);
        return loss;
    }

    public Text getWeightAsText() {
        String ret = "";
        for (int i = 0; i < weightvector.length; i++) {
            if (weightvector[i] != 0) {
                ret = ret + " " + i + ":" + weightvector[i];
            }
        }
        return new Text(ret);
    }
    
}