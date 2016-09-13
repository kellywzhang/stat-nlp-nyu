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
class EmpiricalBigramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static double lambda = 0.9; // weight on bigram (vs unigram)
	
	public static boolean setLambda(double value) {
		if (value < 0 || value > 1) {
			return false;
		}
		lambda = value;
		return true;
	}

	// Counter for first word in bigram (unigrams)
	Counter<String> wordCounter = new Counter<String>();
	// maps first words to counters for second words
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();

	// linear combination of unigram and bigram probabilities
	public double getBigramProbability(String previousWord, String word) {
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}
		return lambda * bigramCount + (1.0 - lambda) * unigramCount;
	}

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

	// unigram method
	String generateWord(String previousWord) {
		double sample = Math.random();
		double sum = 0.0;
		Counter<String> counter = bigramCounter.getCounter(previousWord);
		for (String word : counter.keySet()) {
			sum += counter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	// altered
	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String word = START;
		while (!word.equals(STOP)) {
			sentence.add(word);
			word = generateWord(word);
		}
		sentence.add(word);
		return sentence;
	}

	public EmpiricalBigramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		//this.lambda = lambda;
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String previousWord = stoppedSentence.get(0);
			for (int i = 1; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		// normalizes each Counter in bigramCounter map
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		// normalizes unigram Counter
		wordCounter.normalize();
	}
}
