package org.mlbio.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.mlbio.classifier.WeightParameter;

// Extract the summary information from learned WeightParameters
// Mapper reads the WeightParameters extracts trainTime 
// then writes it to a text file
// the text files can be downloaded using getMerge and the relevant info can be extracted.

public class SummaryExtractor extends Configured implements Tool {

	// Mapper reads the WeightParameters
	public static class ExtractTrainTimeMapper extends
			Mapper<IntWritable, WeightParameter, IntWritable, Text> {

//		private String weightString(WeightParameter param){
//			
//			String rslt = "";
//			double eps = Math.pow(10, -9);
//			for(int i=0;i<param.weightvector.length;i++){
//				if(Math.abs(param.weightvector[i]) >= eps){
//					rslt = rslt + " " + i + ":" + param.weightvector[i];
//				}
//			}
//			return rslt;
//		}
//		
		public void map(IntWritable key, WeightParameter param, Context context)
				throws IOException, InterruptedException {

			// format node_id train_time trainObjective
			String outStr = " " + param.node 
					+ " & " + param.trainTime 
					+ " & " + param.trainObjective;
//			String outStr = weightString(param);
			context.write(key, new Text(outStr));
		}
	}

	public int run(String[] args) throws Exception {
		Configuration conf = getConf();
		String inputDir = conf.get("exinfo.input");
		String outputDir = conf.get("exinfo.output");

		JobConf jconf = new JobConf(conf);
		Job job = new Job(jconf);
		job.setJarByClass(SummaryExtractor.class);
		job.setMapperClass(ExtractTrainTimeMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(inputDir));
		FileOutputFormat.setOutputPath(job, new Path(outputDir));
		if (job.waitForCompletion(true) == false) {
			System.out.println("Job Failed!");
		} else {
			System.out.println("Job Succeeded!");
		}

		return 0;
	}

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new Configuration(), new SummaryExtractor(), args);
	}

}
