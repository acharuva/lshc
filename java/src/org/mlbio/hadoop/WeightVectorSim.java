package org.mlbio.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.mlbio.classifier.WeightParameter;

import java.util.Iterator;

public class WeightVectorSim extends Configured implements Tool {


    public static class ExtractTrainTimeMapper extends
            Mapper<IntWritable, WeightParameter, IntWritable, WeightParameter> {

        public void map(IntWritable key, WeightParameter param, Context context)
                throws IOException, InterruptedException {
        	
        	if(param.isLeaf()){
        		context.write(new IntWritable(param.node), param);
        	}	
        }
    }
    
    public static  class SimReducer extends 
    	Reducer<IntWritable, WeightParameter, IntWritable, Text> {
    	
    	public void reduce(IntWritable key,
				Iterable<WeightParameter> reduceList, Context context)
				throws IOException, InterruptedException {
    	
    		WeightParameter w1 = new WeightParameter(reduceList.iterator().next());
    		WeightParameter w2 = new WeightParameter(reduceList.iterator().next());
    		
    		if(reduceList.iterator().hasNext()){
    			throw new RuntimeException("Iterator has more values than expected. NODE_ID = "+key);
    		}
    		
    		double sim = similarity(w1.weightvector, w2.weightvector);
    		System.out.println(" "+key+": "+ sim);
    		context.write(key, new Text(""+sim));
    	}
    	
    	private double similarity(float[] a, float[] b) {
			// similarity defined as normalized dot product
			double normA = 0;
			double normB = 0;
			double dotProd = 0;
			for (int i = 0; i < a.length; i++) {
				normA += a[i] * a[i];
				normB += b[i] * b[i];
				dotProd += a[i] * b[i];
			}
			double normDotProd = dotProd / Math.sqrt(normA * normB);
			return normDotProd;
		}
    }

    public int run(String[] args) throws Exception {
    	
    	
    	Configuration conf = getConf();
        String inputDir = conf.get("exinfo.input");
        String outputDir = conf.get("exinfo.output");
        
    	System.out.println(inputDir);
    	System.out.println(outputDir);
        
        JobConf jconf = new JobConf(conf);
        Job job = new Job(jconf);
        job.setJarByClass(WeightVectorSim.class);
        
        job.setMapperClass(ExtractTrainTimeMapper.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(WeightParameter.class);

        job.setReducerClass(SimReducer.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        
        job.setInputFormatClass(SequenceFileInputFormat.class);
        FileInputFormat.setInputPaths(job, inputDir);
        
        
        job.setOutputFormatClass(TextOutputFormat.class);
        FileOutputFormat.setOutputPath(job, new Path(outputDir));
        
        
        if (job.waitForCompletion(true) == false) {
            System.out.println("Job Failed!");
        } else {
            System.out.println("Job Succeeded!");
        }

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new Configuration(), new WeightVectorSim(), args);
    }

}
