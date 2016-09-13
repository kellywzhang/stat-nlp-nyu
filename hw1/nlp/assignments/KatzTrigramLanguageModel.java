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
class KatzTrigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static int cutOff = 5; // max value to back off for
	static double lambda1 = 0.28; // trigram weight
	static double lambda2 = 0.64; // bigram weight
	// unigram weight = 0.02

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();
	Counter<String> probabilities = new Counter<String>(); //unigram
	Counter<String> backoffs = new Counter<String>();

	// CHANGE
	public double getTrigramProbability(String prePreviousWord, String previousWord, String word) {

		// trigram probability used
		double trigramProbability = probabilities.getCount(prePreviousWord + " " +previousWord + " " + word);
		double bigramProbability = probabilities.getCount(previousWord + " " + word);
		double unigramProbability = probabilities.getCount(word);

		if (Double.isNaN(trigramProbability)
				|| Double.isInfinite(trigramProbability)
				|| trigramProbability < 0)
			System.err.println("stop!");

		if (trigramProbability != 0)
			return lambda1*trigramProbability + lambda2*bigramProbability + (1-lambda1+lambda2)*unigramProbability;

		// bigram probability used
		if (Double.isNaN(bigramProbability)
				|| Double.isInfinite(bigramProbability)
				|| bigramProbability < 0)
			System.err.println("stop!");

		// multiply terms by alpha?
		double backoff = backoffs.getCount(prePreviousWord + " " + previousWord);
		if (backoff == 0) {
			// new word seen
			if (probabilities.getCount(previousWord) == 0)
				backoff = 1.0;
		}

		if (bigramProbability != 0)
			return lambda2*bigramProbability * backoff + (1-lambda1+lambda2)*unigramProbability;

		// backoff to unigram
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
		backoff *= backoffs.getCount(previousWord);
		if (backoff == 0) {
			// new word seen
			if (probabilities.getCount(previousWord) == 0)
				backoff = 1.0;
		}
		return (1-lambda1+lambda2)*unigramProbability * backoff;
	}

	// CHANGE
	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePreviousWord = stoppedSentence.get(0);
		String previousWord = stoppedSentence.get(1);
		for (int i = 2; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getTrigramProbability(prePreviousWord, previousWord,
					word);
			prePreviousWord = previousWord;
			previousWord = word;
		}
		//if (probability == 0)
		//	System.err.println("Underflow");
		return probability;
	}

	// Change
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

	// Change
	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = generateWord();
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord();
		}
		return sentence;
	}

	public KatzTrigramLanguageModel(Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String prePreviousWord = stoppedSentence.get(0);
			String previousWord = stoppedSentence.get(1);
			for (int i = 2; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				trigramCounter.incrementCount(prePreviousWord + " " + previousWord, word, 1.0);
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		//wordCounter.incrementCount(UNKNOWN, 1.0);
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

		double[] trigramBuckets = new double[cutOff + 2];
		for (String previousWords : trigramCounter.keySet()) {
			Counter<String> currentCounter = trigramCounter.getCounter(previousWords);
			for (String word : currentCounter.keySet()) {
				double count = currentCounter.getCount(word);
				if (count <= cutOff + 1)
					trigramBuckets[(int) count]++;
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


		A = (cutOff + 1) * trigramBuckets[cutOff + 1] / trigramBuckets[1];
		forwardProbability = new Counter<String>();
		backwardProbability = new Counter<String>();

		// Trigram setup: loop over w_(i-2), w_(i-1) 
		for (String previousWords : trigramCounter.keySet()) {
			Counter<String> currentCounter = trigramCounter.getCounter(previousWords);

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
							* trigramBuckets[(int) count + 1]
							/ trigramBuckets[(int) count];
					probability = count * normalizer
							* (discountedCount / count - A) / (1 - A);
				}
				if (Double.isNaN(probability) || Double.isInfinite(probability)
						|| probability < 0)
					System.err.println("stop2");
				probabilities.setCount(previousWords + " " + word, probability);

				// sum of probabilities of bigrams with counts less than k
				backwardProbability.incrementCount(previousWords, probabilities.getCount(word));
				probabilitySoFar += probability;
			}

			// sum of probabilities of bigrams with counts greater than k
			forwardProbability.setCount(previousWords, probabilitySoFar);
		}

		// loop over all words
		for (String word : bigramCounter.keySet()) {

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
