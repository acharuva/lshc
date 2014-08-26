package org.mlbio.classifier;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

/*
 * This implements writable because it has to be written to HDFS filesystem
 * 
 */
public class Example implements Writable {
    public int[] fids;
    public float[] fvals;
    public int[] labels;
    public int docid;

    public final void normalize() {
        //change so that it does not include the bias
        double tot = 0;
        for(int i=1; i<fvals.length;i++){
            float fv = fvals[i];
            tot += fv * fv;
        }
            
        tot = Math.sqrt(tot);
        if (tot > 0) {
            for (int i = 1; i < fvals.length; ++i)
                fvals[i] /= tot;
        }
    }

    public int fsize() {
        if (fids.length > 0)
            return fids[fids.length-1]+1;
        return 0;
    }

    public Example(final Example e) {
        this.docid = e.docid;
        this.labels = e.labels.clone();
        this.fids = e.fids.clone();
        this.fvals = e.fvals.clone();
    }

    public Example() {
        fids = new int[0];
        fvals = new float[0];
        labels = new int[0];
        docid = 0;
    }

    public Example(Text value) {
        fids = new int[0];
        fvals = new float[0];
        labels = new int[0];
        parseString(value.toString());
    }

    public Example(String x) {
        fids = new int[0];
        fvals = new float[0];
        labels = new int[0];
        docid = 0;
        parseString(x);
    }
    
    private void parseString(String s){
        String[] tokens1 = s.split("#");
        if(tokens1.length > 1){
            docid = Integer.parseInt(tokens1[1].trim());
        }
        
        String[] tokens2 = tokens1[0].split(",");
        int numLabels = tokens2.length;
        labels = new int[numLabels];
        for(int i=0;i<numLabels-1;i++){
            labels[i] = Integer.parseInt(tokens2[i].trim());
        }
        
        String[] tokens3 = tokens2[tokens2.length-1].trim().split("\\s+");
        //first token is the remaining label
        labels[numLabels-1] = Integer.parseInt(tokens3[0].trim());
        
        // one extra unit features, 
        // which is placed at index 0
        int numFeatures = tokens3.length;
        fids = new int[numFeatures];
        fvals = new float[numFeatures];
        fids[0] = 0;
        fvals[0] = 1;
        for(int i=1;i<numFeatures;i++){
            String[] tokens4 = tokens3[i].split(":");
            fids[i] = Integer.parseInt(tokens4[0].trim());
            fvals[i] = Float.parseFloat(tokens4[1].trim());
        }
    }
    
    public void print() {
        System.out.print(labels.toString());
        for (int i = 0; i < fids.length; ++i)
            System.out.print("  " + fids[i] + ":" + fvals[i]);
        System.out.println(" # " + docid + "\n");
    }

    public void readFields(DataInput in) throws IOException {
        fids = new int[0];
        fvals = new float[0];
        labels = new int[0];

        int nl = in.readInt();
        labels = new int[nl];
        for (int i = 0; i < nl; ++i) {
            labels[i] = in.readInt();
        }
        int nf = in.readInt();
        fids = new int[nf];
        fvals = new float[nf];

        for (int i = 0; i < nf; ++i) {
            int a = in.readInt();
            float b = in.readFloat();
            fids[i] = a;
            fvals[i] = b;
        }
        docid = in.readInt();
    }

    public void write(DataOutput out) throws IOException {
        out.writeInt(labels.length);
        for (int i = 0; i < labels.length; ++i)
            out.writeInt(labels[i]);

        out.writeInt(fids.length);
        for (int i = 0; i < fids.length; ++i) {
            out.writeInt(fids[i]);
            out.writeFloat(fvals[i]);
        }
        out.writeInt(docid);
    }

    public String toString() {
        String str = "" + labels.toString();
        for (int i = 0; i < fids.length; ++i) {
            str = str + " " + fids[i] + ":" + fvals[i];
        }
        str = str + " # " + docid;
        return str;
    }
}