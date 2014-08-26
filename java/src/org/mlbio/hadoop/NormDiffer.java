package org.mlbio.hadoop;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
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

// Computes the similarity of a node to the mean of its neighbors
// and its parent

// hadoop jar {jar} org.mlbio.hadoop.NormDiffer 
//	-D input=/user/acharuva/lsmtl/output/hrsvmicd/dmoz_2010/C_5/1/weights/
//	-D output=/user/acharuva/lsmtl/output/hrsvmicd/dmoz_2010/C_5/1/wsim/


public class NormDiffer extends Configured implements Tool {
	public static class Map1 extends
			Mapper<IntWritable, WeightParameter, IntWritable, WeightParameter> {

		public void map(IntWritable key, WeightParameter param, Context context)
				throws IOException, InterruptedException {

			if (param.isLeaf()) {
				// if leaf then only neighbor is parent node
				if (param.neighbors.length > 1) {
					throw new RuntimeException(
							"Number of neighbors of leaf node "
									+ "can not exceed 1");
				} else {
					int parent = param.neighbors[0];
					context.write(new IntWritable(parent), param);
				}
			} else {
				context.write(key, param);
			}
		}
	}

	public static class Reduce1 extends
			Reducer<IntWritable, WeightParameter, IntWritable, Text> {

		public void reduce(IntWritable key,
				Iterable<WeightParameter> reduceList, Context context)
				throws IOException, InterruptedException {
			WeightParameter parent = null;
			ArrayList<WeightParameter> children = new ArrayList<WeightParameter>();

			// copy all the children nodes into ArrayList
			int numChildren = 0;
			for (WeightParameter parm : reduceList) {
				if (parm.node == key.get()) {
					parent = new WeightParameter(parm);
				} else {
					System.out.println("parent,child=" + key + "," + parm.node);
					children.add(new WeightParameter(parm));
					numChildren++;
				}
			}

			// compute average of all the children
			float[] average = new float[parent.weightvector.length];
			for (int i = 0; i < average.length; i++) {
				average[i] = 0;
				for (WeightParameter child : children) {
					average[i] += child.weightvector[i];
				}
				average[i]/=numChildren;
			}

			for (WeightParameter child : children) {
				double parSim = similarity(parent.weightvector,
						child.weightvector);
				
				// compute average similarity to all siblings
				double sibSimAve = 0;
				int numSiblings = numChildren - 1;
				for (WeightParameter sib : children) {
					if (sib != child) {
						sibSimAve += similarity(sib.weightvector,
								child.weightvector);
					}
				}
				sibSimAve /= numSiblings;
				
				// compute similarity to average of all siblings
				float[] sibAve = new float[child.weightvector.length];
				for(int i=0;i<sibAve.length;i++){
					sibAve[i] = average[i]*numChildren - child.weightvector[i];
					sibAve[i] /= (numChildren-1); 
				}
				double sibAveSim = similarity(sibAve, child.weightvector);

				context.write(new IntWritable(child.node), new Text(parent.node
						+ " " + numChildren + " " + parSim + " " + sibSimAve + " "
						+ sibAveSim ));
			}

			System.out.println("*****");
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
		String inputDir = conf.get("input");
		String outputDir = conf.get("output");

		JobConf jconf = new JobConf(conf);
		Job job = new Job(jconf);
		job.setJarByClass(SummaryExtractor.class);
		job.setMapperClass(Map1.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(WeightParameter.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(Text.class);
		job.setReducerClass(Reduce1.class);
		job.setNumReduceTasks(30);
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
		ToolRunner.run(new Configuration(), new NormDiffer(), args);
	}

}
