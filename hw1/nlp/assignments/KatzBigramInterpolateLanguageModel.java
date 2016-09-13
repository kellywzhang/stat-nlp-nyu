package nlp.assignments;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import nlp.langmodel.LanguageModel;
import nlp.util.Counter;
import nlp.util.CounterMap;

/**
 * A dummy language model -- uses empirical unigram counts, plus a single
 * ficticious count for unknown words.
 */
class KatzBigramInterpolateLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static int cutOff = 10; // max value to back off for
	static double lambda = 0.94;

	// keys are words; values are their counts (simple unigram)
	Counter<String> wordCounter = new Counter<String>();
	// bigram (keys are words, values are Counters; these counters have words as keys, counts of bigrams as values)
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	Counter<String> probabilities = new Counter<String>();
	Counter<String> backoffs = new Counter<String>();
	// discounted version of WordCounter
	Counter<String> discountedWordCounter = new Counter<String>();
	// counts bigrams; keys are "first_word second_word", values are counts
	Counter<String> discountedBigramCounter = new Counter<String>();

	public static boolean setCutOff(int value) {
		if (value < 0) {
			return false;
		}
		cutOff = value;
		return true;
	}

	public static boolean setLambda(double value) {
		if (value < 0 || value > 1) {
			return false;
		}
		lambda = value;
		return true;
	}

	public double getBigramProbability(String previousWord, String word) {
		
		// bigram probability used
		double bigramProbability = probabilities.getCount(previousWord + " " + word);
		
		if (Double.isNaN(bigramProbability)
				|| Double.isInfinite(bigramProbability)
				|| bigramProbability < 0)
			System.err.println("stop!");

		// backoff to unigram
		double unigramProbability = probabilities.getCount(word);

		if (bigramProbability != 0)
			return lambda*bigramProbability + (1-lambda)*unigramProbability;

		// unseen word
		if (unigramProbability == 0) {
			// System.out.println("UNKNOWN Word: " + word);
			unigramProbability = probabilities.getCount(UNKNOWN);
		}

		if (Double.isNaN(unigramProbability)
				|| Double.isInfinite(unigramProbability)
				|| unigramProbability < 0)
			System.err.println("stop!!");

		// multiply terms by alpha?
		double backoff = backoffs.getCount(previousWord);
		if (backoff == 0) {
			// new word seen
			if (probabilities.getCount(previousWord) == 0)
				backoff = 1.0;
		}
		return (1-lambda)*unigramProbability * backoff;
	}

	// check this
	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String previousWord = stoppedSentence.get(0);
		for (int i = 1; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getBigramProbability(previousWord, word);
			previousWord = word;
		}
		return probability;
	}

	// alter
	String generateWord() {
		double sample = Math.random();
		double sum = 0.0;
		for (String word : wordCounter.keySet()) {
			sum += wordCounter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	// alter
	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	public KatzBigramInterpolateLanguageModel(Collection<List<String>> sentenceCollection) {
		// loop over sentences
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);

			// loop over words in sentence
			for (int i = 1; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);

				// unigram count
				wordCounter.incrementCount(word, 1.0);
				// not used...
				discountedWordCounter.incrementCount(word, 1.0);
				// not used...
				discountedBigramCounter.incrementCount(previousWord + " " + word, 1.0);
				// bigram count
				bigramCounter.incrementCount(previousWord, word, 1.0);
				previousWord = word;
			}
		}
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		// cutoff = k
		// getting counts of unigrams; recording those that are less than cutoff+1
		double[] unigramBuckets = new double[cutOff + 2];
		for (String word : wordCounter.keySet()) {
			double count = wordCounter.getCount(word);
			if (count <= cutOff + 1)
				unigramBuckets[(int) count]++;
		}

		// getting counts of bigrams; recording those that have count less than cutoff+1
		double[] bigramBuckets = new double[cutOff + 2];
		for (String previousWord : bigramCounter.keySet()) {
			Counter<String> currentCounter = bigramCounter
					.getCounter(previousWord);
			for (String word : currentCounter.keySet()) {
				double count = currentCounter.getCount(word);
				if (count <= cutOff + 1)
					bigramBuckets[(int) count]++;
			}
		}

		// 1/N
		double normalizer = 1.0 / wordCounter.totalCount();
		double A = (cutOff + 1) * unigramBuckets[cutOff + 1]
				/ unigramBuckets[1];

		// Unigram setup: loop over all words
		for (String word : wordCounter.keySet()) {
			double count = wordCounter.getCount(word);

			// undiscounted probability if count greater than cutoff
			if (count > cutOff)
				probabilities.setCount(word, count * normalizer);
			else {
				// good-turing count
				double discountedCount = (count + 1)
						* unigramBuckets[(int) count + 1]
						/ unigramBuckets[(int) count];
				double probability = count * normalizer
						* (discountedCount / count - A) / (1 - A);
				probabilities.setCount(word, probability);
				if (Double.isNaN(probability) || Double.isInfinite(probability)
						|| probability < 0)
					System.err.println("stop1");
			}
		}
		// probability of unseen is N_1/N
		probabilities.setCount(UNKNOWN, unigramBuckets[1] * normalizer
				/ wordCounter.size());

		A = (cutOff + 1) * bigramBuckets[cutOff + 1] / bigramBuckets[1];
		Counter<String> forwardProbability = new Counter<String>();
		Counter<String> backwardProbability = new Counter<String>();

		// Bigram setup: loop over w_(i-1) 
		for (String previousWord : bigramCounter.keySet()) {
			Counter<String> currentCounter = bigramCounter.getCounter(previousWord);

			// 1 / count of previousWord
			normalizer = 1.0 / currentCounter.totalCount();
			double probability = 0;
			double probabilitySoFar = 0;

			// loop over w_i
			for (String word : currentCounter.keySet()) {
				// number of times word comes after previousWord
				double count = currentCounter.getCount(word);

				if (count > cutOff) {
					probability = count * normalizer;
					// probability *= 0.99;
				} else {
					// good-turning count
					double discountedCount = (count + 1)
							* bigramBuckets[(int) count + 1]
							/ bigramBuckets[(int) count];
					probability = count * normalizer
							* (discountedCount / count - A) / (1 - A);
				}
				if (Double.isNaN(probability) || Double.isInfinite(probability)
						|| probability < 0)
					System.err.println("stop2");
				probabilities.setCount(previousWord + " " + word, probability);

				// sum of probabilities of bigrams with counts less than k
				backwardProbability.incrementCount(previousWord, probabilities.getCount(word));
				probabilitySoFar += probability;
			}

			// sum of probabilities of bigrams with counts greater than k
			forwardProbability.setCount(previousWord, probabilitySoFar);
		}

		// loop over all words
		for (String word : wordCounter.keySet()) {

			// alpha value
			double backoff = (1.0 - forwardProbability.getCount(word))
					/ (1.0 - backwardProbability.getCount(word));
			if (Double.isNaN(backoff) || Double.isInfinite(backoff)
					|| backoff < 0)
				System.err.println("stop3");
			backoffs.setCount(word, backoff);
		}
	}
}
