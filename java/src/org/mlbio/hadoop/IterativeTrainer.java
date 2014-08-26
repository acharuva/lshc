package org.mlbio.hadoop;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Vector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.mlbio.classifier.BinarySVM;
import org.mlbio.classifier.Example;
import org.mlbio.classifier.WeightParameter;

public class IterativeTrainer extends Configured implements Tool {
	static int NUM_REDUCE_TASKS = 30;
	
	public static class InitMapper extends
			Mapper<LongWritable, Text, IntWritable, WeightParameter> {

		private Vector<Example> data;
		private boolean dataLoaded = false;
		private int featureDimensions = -1;
		
		protected void setup(Context context) throws IOException {
			if (dataLoaded)
				return;
			else
				dataLoaded = true;

			Configuration config = context.getConfiguration();
			System.out.println(" Opening Training Instances From "
					+ config.get("train.dataset"));

			data = new Vector<Example>();
			
			FileSystem fs = FileSystem.getLocal(config);
			
			// DISTCACHE
			Path[] listedPaths = DistributedCache.getLocalCacheFiles(config);
//			Path pattern = new Path(config.get("train.dataset") + "/*");
//			FileStatus[] fStat = fs.globStatus(pattern);
//			Path[] listedPaths = FileUtil.stat2Paths(fStat);

			
			// DISTCAHE END
			

			// Parse the training data
			for (Path p : listedPaths) {
				System.out.println(" Opening Distributed Cache file "
						+ p.toString());
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, p,
						config);
				Writable Key = (Writable) ReflectionUtils.newInstance(
						reader.getKeyClass(), config);
				Writable Value = (Writable) ReflectionUtils.newInstance(
						reader.getValueClass(), config);

				while (reader.next(Key, Value)) {
					Example e = new Example((Example) Value);
					data.add(e);
					featureDimensions = Math.max(featureDimensions, e.fsize());
				}
				reader.close();
			}
			System.out.println(" Loaded " + data.size() + " Instances");
			System.out.println(" Num Features = " + featureDimensions);
		}

		public void map(LongWritable key, Text value, Context context)
				throws IOException, InterruptedException {
			Configuration config = context.getConfiguration();
			double C = config.getFloat("train.svm.C", -1);
			int iter = config.getInt("train.current_iter", -1);
			if (featureDimensions == 0 || C < 0 || iter < 0) {
				throw new IllegalStateException(
						"Invalid value of C or ndim or iter.");
			}

			// read the list of nodes and create empty vectors
			WeightParameter param = WeightParameter.parseString(
					value.toString(), featureDimensions);

			if (param.weightvector.length - 1 != featureDimensions) {
				throw new IllegalArgumentException("node:" + param.node
						+ " length:" + param.weightvector.length + "\n"
						+ value.toString());
			}

			// compute objective function value and write to hdfs
			// norm is all zeros
			double loss = 0;
			if (param.isLeaf()) {
				for (Example e : data) {
					loss += param.hingeLoss(e);
				}
			}

			// The following expression should be
			// 1/2*norm + C* loss
			// but the norm is being calculated in both
			// the nodes involved hence the contribution is
			// halved here.
			double objective = C * loss;
			param.trainObjective = objective;

			try {
				context.write(new IntWritable(param.getNode()), param);
			} catch (IOException e) {
				e.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public static class ConvertTextMapper extends
			Mapper<IntWritable, WeightParameter, IntWritable, Text> {

		public void map(IntWritable key, WeightParameter value, Context context) {
			// convert the WeightParameter to text
			Text weights = value.getWeightAsText();
			try {
				context.write(key, weights);
			} catch (IOException e) {
				e.printStackTrace();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	public static class SVMMapper extends
			Mapper<IntWritable, WeightParameter, IntWritable, WeightParameter> {

		public void map(IntWritable key, WeightParameter param, Context context)
				throws IOException, InterruptedException {

			// emit the weight vector to each neighbor and itself.
			for (int n : param.neighbors) {
				context.write(new IntWritable(n), param);
			}
			context.write(key, param);
		}
	}

	public static class SVMReducer extends
			Reducer<IntWritable, WeightParameter, IntWritable, WeightParameter> {

		private Vector<Example> data;
		private boolean dataLoaded = false;

		protected void setup(Context context) throws IOException {
			if (dataLoaded)
				return;
			else
				dataLoaded = true;

			Configuration conf = context.getConfiguration();
			System.out.println(" Opening Training Instances From "
					+ conf.get("train.dataset"));
			data = new Vector<Example>();
			FileSystem fs = FileSystem.getLocal(conf);
			
			// DISTCACHE
			Path[] listedPaths = DistributedCache.getLocalCacheFiles(conf);
//			Path pattern = new Path(conf.get("train.dataset") + "/*");
//			FileStatus[] fStat = fs.globStatus(pattern);
//			Path[] listedPaths = FileUtil.stat2Paths(fStat);

			
			// DISTCAHE END
			

			// Parse the training data
			for (Path p : listedPaths) {
				System.out.println(" Opening Distributed Cache file "
						+ p.toString());
				SequenceFile.Reader reader = new SequenceFile.Reader(fs, p,
						conf);
				Writable key = (Writable) ReflectionUtils.newInstance(
						reader.getKeyClass(), conf);
				Writable value = (Writable) ReflectionUtils.newInstance(
						reader.getValueClass(), conf);

				while (reader.next(key, value)) {
					data.add(new Example((Example) value));
				}
				reader.close();
			}
			System.out.println(" Loaded " + data.size() + " Instances");
		}

		public void reduce(IntWritable key,
				Iterable<WeightParameter> reduceValList, Context context)
				throws InterruptedException, IOException {
			// aggregates the weight parameters and trains a model

			long startTime = System.nanoTime();
			Configuration config = context.getConfiguration();
			
			double smvC = config.getFloat("train.svm.C", 1);
			double eps = config.getFloat("train.svm.eps", (float) 0.001);
			int max_iter = config.getInt("train.svm.max_iter", 100);
			int currentIter = config.getInt("train.current_iter", -1);
			int numTrainSets = config.getInt("train.num_train_sets", -1);
			WeightParameter thisNodeParam = null;
			ArrayList<WeightParameter> neighborParamList = new ArrayList<WeightParameter>();

			if (currentIter < 0) {
				throw new IllegalArgumentException("Invalid current_iter.");
			}
			if (numTrainSets < 0) {
				throw new IllegalArgumentException("Invalid num_train_sets.");
			}

			for (WeightParameter par : reduceValList) {
				if (par.node == key.get())
					thisNodeParam = new WeightParameter(par);
				else
					neighborParamList.add(new WeightParameter(par));
			}

			// Self weights assignment check
			if (thisNodeParam == null) {
				throw new IllegalArgumentException(
						"reduceValList does not have self key:" + key.get());
			} else if (thisNodeParam.node != key.get()) {
				throw new IllegalStateException("the assignment for "
						+ key.get() + " is incorrectly done to "
						+ thisNodeParam.node);
			}

			// Leaf node can not have more than one neighbors for a Tree
			if (thisNodeParam.isLeaf() && neighborParamList.size() > 1) {
				throw new IllegalStateException(
						"Leaf Node has more than one Neighbors");
			}

			// Set by set training.
			// if numTrainSets == 1 all nodes are modified in every iteration.
			if (numTrainSets == 1
					|| thisNodeParam.trainSetNum == (currentIter - 1)
							% numTrainSets) {
				int numNeighbors = neighborParamList.size();
				// compute mean weight vector
				for (int i = 0; i < thisNodeParam.weightvector.length; i++) {
					thisNodeParam.weightvector[i] = 0;
					for (WeightParameter par : neighborParamList) {
						thisNodeParam.weightvector[i] += par.weightvector[i];
					}
					thisNodeParam.weightvector[i] /= numNeighbors;
				}

				// if leaf node then train otherwise just output the average
				if (thisNodeParam.isLeaf()) {
					BinarySVM svm = new BinarySVM();
					svm.optimize(data, thisNodeParam, smvC, eps, max_iter);
				}
				if (thisNodeParam.node != key.get()) {
					throw new IllegalArgumentException("Key changed from "
							+ key.get() + " to " + thisNodeParam.node);
				}
			}

			long endTime = System.nanoTime();
			thisNodeParam.trainTime = (endTime - startTime)/ Math.pow(10, 9);
			
			
			startTime = System.nanoTime();
			// compute and save objective function value
			// to WeightParam
			double normL2Square = 0;
			for (WeightParameter p : neighborParamList) {
				normL2Square += thisNodeParam.normDiff(p);
			}

			double loss = 0;
			if (thisNodeParam.isLeaf()) {
				for (Example e : data) {
					loss += thisNodeParam.hingeLoss(e);
				}
			}			  			
			
			// The following expression should be
			// 1/2*norm + smvC* loss
			// but the norm is being calculated in both
			// the nodes involved hence the contribution is
			// halved here.
			double objective = normL2Square / 4.0 + smvC * loss;
			thisNodeParam.trainObjective = objective;
			endTime = System.nanoTime();
			thisNodeParam.objCalcTime = Math.round(10 * (endTime - startTime)/ Math.pow(10, 9)) / 10.0;
			
			// output learned weights
			context.write(key, thisNodeParam);
		}
	}

	public static Configuration addPathToDC(Configuration conf, String path)
			throws IOException {
		// add files to distributed cache
		FileSystem fs = FileSystem.get(conf);
		FileStatus[] fstatus = fs.globStatus(new Path(path));
		Path[] listedPaths = FileUtil.stat2Paths(fstatus);
		for (Path p : listedPaths) {
			System.out.println(" Add File to DC " + p.toUri().toString());
			DistributedCache.addCacheFile(p.toUri(), conf);
		}
		return conf;
	}

	public void doInitJob(Configuration conf) throws IOException,
			InterruptedException, ClassNotFoundException {

		// Initializes weight vectors to 0 vectors, before
		// before, the first iteration.

		String output = conf.get("train.output");
		String input = conf.get("train.input");
		String outputDir = output + "/" + "0/weights/";

		Job job = new Job(conf);
		job.setJarByClass(IterativeTrainer.class);
		job.setJobName("convert");
		job.setMapperClass(InitMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(WeightParameter.class);
		job.setMapOutputValueClass(WeightParameter.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, input);
		SequenceFileOutputFormat.setCompressOutput(job, true);
		FileOutputFormat.setOutputPath(job, new Path(outputDir));
		if (job.waitForCompletion(true) == false) {
			System.out.println("Job Failed!");
		} else {
			System.out.println("Job Succeeded!");
		}
	}

	public void doConvertWeightsToText(Configuration conf) throws IOException,
			InterruptedException, ClassNotFoundException {

		// Converts the output weights
		// from SequenceFile format to Text format
		// typically used for the outputs of last iteration

		String workDir = conf.get("train.output");
		int iter = conf.getInt("train.current_iter", -1);
		if (iter < 0) {
			throw new IllegalStateException(
					"Invalid value of current Iteration number.");
		}
		String input = workDir + "/" + iter + "/weights/";
		String outputDir = workDir + "/" + iter + "/text_weights/";

		Job job = new Job(conf);
		job.setJarByClass(IterativeTrainer.class);
		job.setJobName("convertToTextWeights");
		job.setMapperClass(ConvertTextMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(IntWritable.class);
		job.setMapOutputValueClass(Text.class);
		job.setNumReduceTasks(0);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);
		FileInputFormat.setInputPaths(job, input);
		FileOutputFormat.setOutputPath(job, new Path(outputDir));
		if (job.waitForCompletion(true) == false) {
			System.out.println("Job Failed!");
		} else {
			System.out.println("Job Succeeded!");
		}

	}

	public void doLearnJob(Configuration conf) throws IOException,
			InterruptedException, ClassNotFoundException {

		// Performs one iteration of HRSVM
		// Reads previous iter weights and writes next iter weights.
		// Repeatedly called for a certain number of iteration or until
		// convergence.

		int iter = conf.getInt("train.current_iter", -1);
		if (iter < 1) {
			throw new IllegalStateException("Invalid iteration number.");
		}
		String workDir = conf.get("train.output");
		String input = workDir + (iter - 1) + "/" + "/weights/";
		String output = workDir + iter + "/" + "/weights/";
//		int numReduceTasks = conf.getInt("train.num_reduce_tasks", 30);

		JobConf jconf = new JobConf(conf);
		jconf.setNumTasksToExecutePerJvm(-1);

		Job job = new Job(jconf);
		job.setJarByClass(IterativeTrainer.class);
		job.setJobName("train-lsmtl");
		job.setMapperClass(SVMMapper.class);
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(WeightParameter.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(WeightParameter.class);
		job.setReducerClass(SVMReducer.class);
		job.setNumReduceTasks(NUM_REDUCE_TASKS);
		job.setInputFormatClass(SequenceFileInputFormat.class);
		job.setOutputFormatClass(SequenceFileOutputFormat.class);
		FileInputFormat.setInputPaths(job, new Path(input));
		SequenceFileOutputFormat.setCompressOutput(job, true);
		FileOutputFormat.setOutputPath(job, new Path(output));
		if (job.waitForCompletion(true) == false) {
			System.out.println("Job Failed!");
		} else {
			System.out.println("Job Succeeded!");
		}
	}
	
	
//	public void printConfig(Configuration config){
//		String[] props = {"mapreduce.tasktracker.map.tasks.maximum", 
//				"train.dataset"
//				};
//		for(int i=0;i<props.length; i++){
//			System.out.println( props[i] + " " +  
//					config.get(props[i]));
//		}
//		
//	}

	public int run(String[] args) throws Exception {
		Configuration conf = new Configuration(getConf());
		conf.setBoolean("mapreduce.fileoutputcommitter.marksuccessfuljobs",
				false);
//		conf.setInt("mapreduce.task.io.sort.mb", 2000);
//		printConfig(conf);
		String dataset = conf.get("train.dataset");
		int numIter = conf.getInt("train.iterations", 1);
		conf = addPathToDC(conf, dataset + "*");
		FileSystem localFS = FileSystem.getLocal(conf);
//		System.out.println(conf.get("train.timeoutput"));
//		System.out.println(conf.get("train.dataset"));
		Path timesOutPath = new Path(conf.get("train.timeoutput"));
		
		FSDataOutputStream out = localFS.create(timesOutPath);
		
		for (int iter = 0; iter <= numIter; iter++) {
			conf.setInt("train.current_iter", iter);

			long startTime = System.nanoTime();
			if (iter == 0)
				doInitJob(conf);
			else
				doLearnJob(conf);
			long endTime = System.nanoTime();
			
			double runTime = Math.round(10 * (endTime - startTime)/ Math.pow(10, 9)) / 10.0;
			System.out.println("Train time Iteration " + iter + ": " + runTime);
			out.writeChars(timesOutPath + " & iteration= " + iter + " & " + runTime + "\n");			
		}
		out.close();

		return 0;
	}

	public static void main(String[] args) throws Exception {
		ToolRunner.run(new Configuration(), new IterativeTrainer(), args);
	}

}
