package nlp.assignments;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.text.NumberFormat;
import java.text.DecimalFormat;

import java.io.File;
import java.io.PrintWriter;

import nlp.langmodel.LanguageModel;
import nlp.util.CommandLineUtils;

/**
 * This is the main harness for assignment 1. To run this harness, use
 * <p/>
 * java nlp.assignments.LanguageModelTester -path ASSIGNMENT_DATA_PATH -model
 * MODEL_DESCRIPTOR_STRING
 * <p/>
 * First verify that the data can be read on your system. Second, find the point
 * in the main method (near the bottom) where an EmpiricalUnigramLanguageModel
 * is constructed. You will be writing new implementations of the LanguageModel
 * interface and constructing them there.
 */
public class LanguageModelTester {

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class EditDistance {
		static double INSERT_COST = 1.0;
		static double DELETE_COST = 1.0;
		static double SUBSTITUTE_COST = 1.0;

		private double[][] initialize(double[][] d) {
			for (int i = 0; i < d.length; i++) {
				for (int j = 0; j < d[i].length; j++) {
					d[i][j] = Double.NaN;
				}
			}
			return d;
		}

		public double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList) {
			double[][] bestDistances = initialize(new double[firstList.size() + 1][secondList
					.size() + 1]);
			return getDistance(firstList, secondList, 0, 0, bestDistances);
		}

		private double getDistance(List<? extends Object> firstList,
				List<? extends Object> secondList, int firstPosition,
				int secondPosition, double[][] bestDistances) {
			if (firstPosition > firstList.size()
					|| secondPosition > secondList.size())
				return Double.POSITIVE_INFINITY;
			if (firstPosition == firstList.size()
					&& secondPosition == secondList.size())
				return 0.0;
			if (Double.isNaN(bestDistances[firstPosition][secondPosition])) {
				double distance = Double.POSITIVE_INFINITY;
				distance = Math.min(
						distance,
						INSERT_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition,
										bestDistances));
				distance = Math.min(
						distance,
						DELETE_COST
								+ getDistance(firstList, secondList,
										firstPosition, secondPosition + 1,
										bestDistances));
				distance = Math.min(
						distance,
						SUBSTITUTE_COST
								+ getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
				if (firstPosition < firstList.size()
						&& secondPosition < secondList.size()) {
					if (firstList.get(firstPosition).equals(
							secondList.get(secondPosition))) {
						distance = Math.min(
								distance,
								getDistance(firstList, secondList,
										firstPosition + 1, secondPosition + 1,
										bestDistances));
					}
				}
				bestDistances[firstPosition][secondPosition] = distance;
			}
			return bestDistances[firstPosition][secondPosition];
		}
	}

	// HELPER CLASS FOR THE HARNESS, CAN IGNORE
	static class SentenceCollection extends AbstractCollection<List<String>> {
		static class SentenceIterator implements Iterator<List<String>> {

			BufferedReader reader;

			public boolean hasNext() {
				try {
					return reader.ready();
				} catch (IOException e) {
					return false;
				}
			}

			public List<String> next() {
				try {
					String line = reader.readLine();
					String[] words = line.split("\\s+");
					List<String> sentence = new ArrayList<String>();
					for (int i = 0; i < words.length; i++) {
						String word = words[i];
						sentence.add(word.toLowerCase());
					}
					return sentence;
				} catch (IOException e) {
					throw new NoSuchElementException();
				}
			}

			public void remove() {
				throw new UnsupportedOperationException();
			}

			public SentenceIterator(BufferedReader reader) {
				this.reader = reader;
			}
		}

		String fileName;

		public Iterator<List<String>> iterator() {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(
						fileName));
				return new SentenceIterator(reader);
			} catch (FileNotFoundException e) {
				throw new RuntimeException("Problem with SentenceIterator for "
						+ fileName);
			}
		}

		public int size() {
			int size = 0;
			Iterator<List<String>> i = iterator();
			while (i.hasNext()) {
				size++;
				i.next();
			}
			return size;
		}

		public SentenceCollection(String fileName) {
			this.fileName = fileName;
		}

		public static class Reader {
			static Collection<List<String>> readSentenceCollection(
					String fileName) {
				return new SentenceCollection(fileName);
			}
		}

	}

	// look at this!
	static double calculatePerplexity(LanguageModel languageModel,
			Collection<List<String>> sentenceCollection) {
		double logProbability = 0.0;
		double numSymbols = 0.0;
		for (List<String> sentence : sentenceCollection) {
			logProbability += Math.log(languageModel
					.getSentenceProbability(sentence)) / Math.log(2.0);
			numSymbols += sentence.size();
		}
		double avgLogProbability = logProbability / numSymbols; // cross-entropy approx
		double perplexity = Math.pow(0.5, avgLogProbability); // why square root of 2, not N?
		return perplexity;
	}

	static double calculateWordErrorRate(LanguageModel languageModel,
			List<SpeechNBestList> speechNBestLists, boolean verbose, boolean errors) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			List<String> bestGuess = null;
			double bestScore = Double.NEGATIVE_INFINITY;
			double numWithBestScores = 0.0;
			double distanceForBestScores = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double score = Math.log(languageModel
						.getSentenceProbability(guess))
						+ (speechNBestList.getAcousticScore(guess) / 16.0);
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (score == bestScore) {
					numWithBestScores += 1.0;
					distanceForBestScores += distance;
				}
				if (score > bestScore || bestGuess == null) {
					bestScore = score;
					bestGuess = guess;
					distanceForBestScores = distance;
					numWithBestScores = 1.0;
				}
			}
			// double distance = editDistance.getDistance(correctSentence,
			// bestGuess);
			totalDistance += distanceForBestScores / numWithBestScores;
			totalWords += correctSentence.size();
			if (verbose) {
				System.out.println();
				displayHypothesis("GUESS:", bestGuess, speechNBestList,
						languageModel);
				displayHypothesis("GOLD:", correctSentence, speechNBestList,
						languageModel);
			} else if (errors) {
				if (bestGuess != correctSentence) {
					System.out.println();
					displayHypothesis("GUESS:", bestGuess, speechNBestList,
							languageModel);
					displayHypothesis("GOLD:", correctSentence, speechNBestList,
							languageModel);
				}
			}
		}
		return totalDistance / totalWords;
	}

	private static NumberFormat nf = new DecimalFormat("0.00E00");

	private static void displayHypothesis(String prefix, List<String> guess,
			SpeechNBestList speechNBestList, LanguageModel languageModel) {
		double acoustic = speechNBestList.getAcousticScore(guess) / 16.0;
		double language = Math.log(languageModel.getSentenceProbability(guess));
		System.out.println(prefix + "\tAM: " + nf.format(acoustic) + "\tLM: "
				+ nf.format(language) + "\tTotal: "
				+ nf.format(acoustic + language) + "\t" + guess);
	}

	// understand what this does
	static double calculateWordErrorRateLowerBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		// what is this?
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double bestDistance = Double.POSITIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance < bestDistance)
					bestDistance = distance;
			}
			totalDistance += bestDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateUpperBound(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double worstDistance = Double.NEGATIVE_INFINITY;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				if (distance > worstDistance)
					worstDistance = distance;
			}
			totalDistance += worstDistance;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static double calculateWordErrorRateRandomChoice(
			List<SpeechNBestList> speechNBestLists) {
		double totalDistance = 0.0;
		double totalWords = 0.0;
		EditDistance editDistance = new EditDistance();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			List<String> correctSentence = speechNBestList.getCorrectSentence();
			double sumDistance = 0.0;
			double numGuesses = 0.0;
			for (List<String> guess : speechNBestList.getNBestSentences()) {
				double distance = editDistance.getDistance(correctSentence,
						guess);
				sumDistance += distance;
				numGuesses += 1.0;
			}
			totalDistance += sumDistance / numGuesses;
			totalWords += correctSentence.size();
		}
		return totalDistance / totalWords;
	}

	static Collection<List<String>> extractCorrectSentenceList(
			List<SpeechNBestList> speechNBestLists) {
		Collection<List<String>> correctSentences = new ArrayList<List<String>>();
		for (SpeechNBestList speechNBestList : speechNBestLists) {
			correctSentences.add(speechNBestList.getCorrectSentence());
		}
		return correctSentences;
	}

	static Set<String> extractVocabulary(
			Collection<List<String>> sentenceCollection) {
		Set<String> vocabulary = new HashSet<String>();
		for (List<String> sentence : sentenceCollection) {
			for (String word : sentence) {
				vocabulary.add(word);
			}
		}
		return vocabulary;
	}


	// Set up default parameters and settings
	static String basePath = ".";
	static String model = "baseline";
	static boolean verbose = false;
	static boolean errors = false;
	static boolean sentences = false;
	static boolean csv = false;

	static List<SpeechNBestList> speechNBestLists;

	// for different values of lambda
	static void runBigram(double min, double max, double step, LanguageModel languageModel) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File("/Users/Kelly/Documents/StatNLP/stat-nlp-nyu/hw1/bigram_output.csv"));
		pw.write("Model,HUB Perplexity,HUB WER,Parameters\n");

		EmpiricalBigramLanguageModel bigram = (EmpiricalBigramLanguageModel) languageModel;
		double counter = min;
		while (counter < max) {
			pw.write("bigram,");
			bigram.setLambda(counter);
			System.out.println("Lambda: " + EmpiricalBigramLanguageModel.lambda);
			double[] toReturn = runModel(bigram);
			pw.write(toReturn[0]+","+toReturn[1]);
			pw.write(",Lambda: " +EmpiricalBigramLanguageModel.lambda + "\n");
			counter += step;
			System.out.println();
		}
		pw.close();
	}

	static void runTrigram(double min1, double max1, double min2, double max2, double step, LanguageModel languageModel) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File("/Users/Kelly/Documents/StatNLP/stat-nlp-nyu/hw1/trigram_output.csv"));
		pw.write("Model,HUB Perplexity,HUB WER,Parameters\n");

		EmpiricalTrigramLanguageModel trigram = (EmpiricalTrigramLanguageModel) languageModel;
		double counter1 = min1;
		double counter2 = min2;
		while (counter1 < max1 && counter1+counter2 < 1) {
			trigram.setLambda1(counter1);
			while (counter2 < max2 && counter1+counter2 < 1) {
				pw.write("trigram,");
				trigram.setLambda2(counter2);
				System.out.println("Lambda1: " + EmpiricalTrigramLanguageModel.lambda1);
				System.out.println("Lambda2: " + EmpiricalTrigramLanguageModel.lambda2);
				double[] toReturn = runModel(trigram);
				pw.write(toReturn[0]+","+toReturn[1]);
				pw.write(",Lambda1: " + EmpiricalTrigramLanguageModel.lambda1 + " Lambda2: " + EmpiricalTrigramLanguageModel.lambda2 +"\n");
				counter2 += step;
				System.out.println();
			}
			counter1 += step;
			counter2 = min2;
			System.out.println();
		}


		counter1 = min1;
		counter2 = min2;
		while (counter1 < max1 && counter1+counter2 < 1) {
			trigram.setLambda2(counter1);
			while (counter2 < max2 && counter1+counter2 < 1) {
				pw.write("trigram,");
				trigram.setLambda1(counter2);
				System.out.println("Lambda1: " + EmpiricalTrigramLanguageModel.lambda1);
				System.out.println("Lambda2: " + EmpiricalTrigramLanguageModel.lambda2);
				double[] toReturn = runModel(trigram);
				pw.write(toReturn[0]+","+toReturn[1]);
				pw.write(",Lambda1: " + EmpiricalTrigramLanguageModel.lambda1 + " Lambda2: " + EmpiricalTrigramLanguageModel.lambda2 +"\n");
				counter2 += step;
				System.out.println();
			}
			counter1 += step;
			counter2 = min2;
			System.out.println();
		}
		pw.close();
	}

	static void runQuadgram(double min1, double max1, double min2, double max2, double min3, double max3, double step, LanguageModel languageModel) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File("/Users/Kelly/Documents/StatNLP/stat-nlp-nyu/hw1/quadgram_output.csv"));
		pw.write("Model,HUB Perplexity,HUB WER,Parameters\n");

		EmpiricalQuadgramLanguageModel quadgram = (EmpiricalQuadgramLanguageModel) languageModel;
		double counter1 = min1;
		double counter2 = min2;
		double counter3 = min3;

		while (counter3 < max3 && counter1+counter2+counter3 < 1) {
			quadgram.setLambda1(counter3);
			while (counter2 < max2 && counter1+counter2+counter3 < 1) {
				quadgram.setLambda2(counter2);
				while (counter1 < max1 && counter1+counter2+counter3 < 1) {
					pw.write("quadgram,");
					quadgram.setLambda3(counter1);
					System.out.println("Lambda1: " + EmpiricalQuadgramLanguageModel.lambda1);
					System.out.println("Lambda2: " + EmpiricalQuadgramLanguageModel.lambda2);
					System.out.println("Lambda3: " + EmpiricalQuadgramLanguageModel.lambda3);
					double[] toReturn = runModel(quadgram);
					pw.write(toReturn[0]+","+toReturn[1]);
					pw.write(",Lambda1: " + EmpiricalQuadgramLanguageModel.lambda1 + " Lambda2: " + EmpiricalQuadgramLanguageModel.lambda2 + " Lambda3: " + EmpiricalQuadgramLanguageModel.lambda3 +"\n");
					counter1 += step;
					System.out.println();
				}
				counter2 += step;
				counter1 = min1;
				System.out.println();
			}
			counter3 += step;
			counter2 = min2;
			counter1 = min1;


			System.out.println();
		}
		pw.close();
	}

	static void runKatzBigram(int min, int max, LanguageModel languageModel) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File("/Users/Kelly/Documents/StatNLP/stat-nlp-nyu/hw1/katzbigram_output.csv"));
		pw.write("Model,HUB Perplexity,HUB WER,Parameters\n");

		KatzBigramLanguageModel katzbigram = (KatzBigramLanguageModel) languageModel;
		int counter = min;
		while (counter < max) {
			pw.write("katz-bigram,");
			KatzBigramLanguageModel.setCutOff(counter);
			System.out.println("cutOff: " + KatzBigramLanguageModel.cutOff);
			double[] toReturn = runModel(katzbigram);
			pw.write(toReturn[0]+","+toReturn[1]);
			pw.write(",cutoff: " + KatzBigramLanguageModel.cutOff + "\n");
			counter++;
			System.out.println();
		}
		pw.close();
	}

	// for different values of lambda
	static void runKatzBigramInterpolate(double min, double max, double step, LanguageModel languageModel) throws FileNotFoundException {
		PrintWriter pw = new PrintWriter(new File("/Users/Kelly/Documents/StatNLP/stat-nlp-nyu/hw1/katz_bigram_interpolate_output.csv"));
		pw.write("Model,HUB Perplexity,HUB WER,Parameters\n");

		KatzBigramInterpolateLanguageModel bigram = (KatzBigramInterpolateLanguageModel) languageModel;
		double counter = min;
		while (counter < max) {
			pw.write("bigram,");
			bigram.setLambda(counter);
			System.out.println("Lambda: " + KatzBigramInterpolateLanguageModel.lambda);
			double[] toReturn = runModel(bigram);
			pw.write(toReturn[0]+","+toReturn[1]);
			pw.write(",Lambda: " +KatzBigramInterpolateLanguageModel.lambda + "\n");
			counter += step;
			System.out.println();
		}
		pw.close();
	}

	public static double[] runModel(LanguageModel languageModel) {
		// Evaluate the language model
		//double wsjPerplexity = calculatePerplexity(languageModel,
		//		testSentenceCollection);
		double hubPerplexity = calculatePerplexity(languageModel,
				extractCorrectSentenceList(speechNBestLists));
		//System.out.println("WSJ Perplexity:  " + wsjPerplexity);
		System.out.println("HUB Perplexity:  " + hubPerplexity);
		/*System.out.println("WER Baselines:");
		System.out.println("  Best Path:  "
				+ calculateWordErrorRateLowerBound(speechNBestLists));
		System.out.println("  Worst Path: "
				+ calculateWordErrorRateUpperBound(speechNBestLists));
		System.out.println("  Avg Path:   "
				+ calculateWordErrorRateRandomChoice(speechNBestLists));*/
		double wordErrorRate = calculateWordErrorRate(languageModel,
				speechNBestLists, verbose, errors);
		System.out.println("HUB Word Error Rate: " + wordErrorRate);

		//System.out.println("\nTest set");
		//double wordErrorRateTest = calculateWordErrorRate(languageModel, speechNBestListsTest, verbose);
		//System.out.println("HUB Word Error Rate Test: " + wordErrorRateTest);

		if (sentences) {
			System.out.println("\nGenerated Sentences:");
			for (int i = 0; i < 10; i++)
				System.out.println("  " + languageModel.generateSentence());
		}

		double[] toReturn = new double[2];
		toReturn[0] = hubPerplexity;
		toReturn[1] = wordErrorRate;
		return toReturn;
	}

	public static void main(String[] args) throws IOException {
		// Parse command line flags and arguments
		Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);

		// Update defaults using command line specifications

		// The path to the assignment data
		if (argMap.containsKey("-path")) {
			basePath = argMap.get("-path");
		}
		System.out.println("Using base path: " + basePath);

		// A string descriptor of the model to use
		if (argMap.containsKey("-model")) {
			model = argMap.get("-model");
		}
		System.out.println("Using model: " + model);

		// Whether or not to print the individual speech errors.
		if (argMap.containsKey("-verbose")) {
			verbose = true;
		}
		if (argMap.containsKey("-quiet")) {
			verbose = false;
		}

		if (argMap.containsKey("-errors")) {
			errors = true;
		}

		if (argMap.containsKey("-sentences")) {
			sentences = true;
		}

		if (argMap.containsKey("-csv")) {
			csv = true;
		}

		// Read in all the assignment data
		String trainingSentencesFile = "/treebank-sentences-spoken.txt";
		String speechNBestListsPath = "/wsj_n_bst";
		Collection<List<String>> trainingSentenceCollection = 
		SentenceCollection.Reader.readSentenceCollection(basePath + trainingSentencesFile);
		Set<String> trainingVocabulary = extractVocabulary(trainingSentenceCollection);
		speechNBestLists = SpeechNBestList.Reader.readSpeechNBestLists(basePath + speechNBestListsPath, trainingVocabulary);

		/*String validationSentencesFile = "/treebank-sentences-spoken-validate.txt";
		Collection<List<String>> validationSentenceCollection =
		SentenceCollection.Reader.readSentenceCollection(basePath + validationSentencesFile);

		String testSentencesFile = "/treebank-sentences-spoken-test.txt";
		Collection<List<String>> testSentenceCollection =
		SentenceCollection.Reader.readSentenceCollection(basePath + testSentencesFile);*/


		// Build the language model
		LanguageModel languageModel = null;
		if (model.equalsIgnoreCase("baseline")) {
			languageModel = new EmpiricalUnigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("sri")) {
			languageModel = new SriLanguageModel(argMap.get("-sri"));
		} else if (model.equalsIgnoreCase("bigram")) {
			languageModel = new EmpiricalBigramLanguageModel(
					trainingSentenceCollection);
			System.out.println("Lambda = " + EmpiricalBigramLanguageModel.lambda);
		} else if (model.equalsIgnoreCase("trigram")) {
			languageModel = new EmpiricalTrigramLanguageModel(
					trainingSentenceCollection);
			System.out.println("Lambda1 = " + EmpiricalTrigramLanguageModel.lambda1);
			System.out.println("Lambda2 = " + EmpiricalTrigramLanguageModel.lambda2);
		} else if (model.equalsIgnoreCase("quadgram")) {
			languageModel = new EmpiricalQuadgramLanguageModel(
					trainingSentenceCollection);
			System.out.println("Lambda1 = " + EmpiricalQuadgramLanguageModel.lambda1);
			System.out.println("Lambda2 = " + EmpiricalQuadgramLanguageModel.lambda2);
			System.out.println("Lambda3 = " + EmpiricalQuadgramLanguageModel.lambda3);
		} else if (model.equalsIgnoreCase("katz-bigram")) {
			languageModel = new KatzBigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-trigram")) {
			languageModel = new KatzTrigramLanguageModel(
					trainingSentenceCollection);
		} else if (model.equalsIgnoreCase("katz-bigram-interpolate")) {
			languageModel = new KatzBigramInterpolateLanguageModel(
					trainingSentenceCollection);
		}
		else {
			throw new RuntimeException("Unknown model descriptor: " + model);
		}

		runModel(languageModel);

		if (csv) {
			if (model.equalsIgnoreCase("bigram")) {
				runBigram(0, 1, 0.05, languageModel);
			} else if (model.equalsIgnoreCase("trigram")){
				runTrigram(0.2, 0.4, 0.5, 0.7, 0.01, languageModel);
			} else if (model.equalsIgnoreCase("quadgram")){
				runQuadgram(0.2, 0.4, 0.15, 0.3, 0.35, 0.5, 0.01, languageModel);
			} else if (model.equalsIgnoreCase("katz-bigram")) {
				runKatzBigram(0, 10, languageModel);
			} else if (model.equalsIgnoreCase("katz-bigram-interpolate")) {
				runKatzBigramInterpolate(0.6, 1, 0.01, languageModel);
			}
		}
	}
}
