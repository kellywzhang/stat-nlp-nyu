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
class EmpiricalQuadgramLanguageModel implements LanguageModel {

	static final String START = "<S>";
	static final String STOP = "</S>";
	static final String UNKNOWN = "*UNKNOWN*";
	static double lambda1 = 0.35; // quadgram weight
	static double lambda2 = 0.21; // trigram weight
	static double lambda3 = 0.39; // bigram weight

	//35, 21, 39

	Counter<String> wordCounter = new Counter<String>();
	CounterMap<String, String> bigramCounter = new CounterMap<String, String>();
	// simply concatenate prePrevious and previous words to denote previous 2 words
	CounterMap<String, String> trigramCounter = new CounterMap<String, String>();
	CounterMap<String, String> quadgramCounter = new CounterMap<String, String>();

	public static boolean setLambda1(double value) {
		if (value < 0 || value > 1) {
			return false;
		}
		lambda1 = value;
		return true;
	}

	public static boolean setLambda2(double value) {
		if (value < 0 || value > 1) {
			return false;
		}
		lambda2 = value;
		return true;
	}

	public static boolean setLambda3(double value) {
		if (value < 0 || value > 1) {
			return false;
		}
		lambda3 = value;
		return true;
	}

	public double getQuadgramProbability(String prePrePreviousWord, String prePreviousWord,
			String previousWord, String word) {
		double quadgramCount = quadgramCounter.getCount(prePrePreviousWord + " " + prePreviousWord + " " + previousWord, word);
		double trigramCount = trigramCounter.getCount(prePreviousWord + " " + previousWord, word);
		double bigramCount = bigramCounter.getCount(previousWord, word);
		double unigramCount = wordCounter.getCount(word);
		if (unigramCount == 0) {
			System.out.println("UNKNOWN Word: " + word);
			unigramCount = wordCounter.getCount(UNKNOWN);
		}
		return lambda1 * quadgramCount + lambda2 * trigramCount + lambda3 * bigramCount
				+ (1.0 - lambda1 - lambda2 - lambda3) * unigramCount;
	}

	public double getSentenceProbability(List<String> sentence) {
		List<String> stoppedSentence = new ArrayList<String>(sentence);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(0, START);
		stoppedSentence.add(STOP);
		double probability = 1.0;
		String prePrePreviousWord = stoppedSentence.get(0);
		String prePreviousWord = stoppedSentence.get(1);
		String previousWord = stoppedSentence.get(2);
		for (int i = 3; i < stoppedSentence.size(); i++) {
			String word = stoppedSentence.get(i);
			probability *= getQuadgramProbability(prePrePreviousWord, prePreviousWord, previousWord,
					word);
			prePrePreviousWord = prePreviousWord;
			prePreviousWord = previousWord;
			previousWord = word;
		}
		return probability;
	}

	String generateWord(String prePrePreviousWord, String prePreviousWord, String previousWord) {
		double sample = Math.random();
		double sum = 0.0;
		Counter<String> counter = quadgramCounter.getCounter(prePrePreviousWord + " " + prePreviousWord + " " + previousWord);
		for (String word : counter.keySet()) {
			sum += counter.getCount(word);
			if (sum > sample) {
				return word;
			}
		}
		return UNKNOWN;
	}

	public List<String> generateSentence() {
		List<String> sentence = new ArrayList<String>();
		String prePrePreviousWord = START;
		String prePreviousWord = START;
		String previousWord = START;
		String next;
		sentence.add(prePrePreviousWord);
		sentence.add(prePreviousWord);
		while (!previousWord.equals(STOP)) {
			sentence.add(previousWord);
			next = generateWord(prePrePreviousWord, prePreviousWord, previousWord);
			prePrePreviousWord = prePreviousWord;
			prePreviousWord = previousWord;
			previousWord = next;
		}
		sentence.add(previousWord);
		return sentence;
	}

	public EmpiricalQuadgramLanguageModel(
			Collection<List<String>> sentenceCollection) {
		for (List<String> sentence : sentenceCollection) {
			List<String> stoppedSentence = new ArrayList<String>(sentence);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);
			stoppedSentence.add(0, START);
			stoppedSentence.add(STOP);
			String prePrePreviousWord = stoppedSentence.get(0);
			String prePreviousWord = stoppedSentence.get(1);
			String previousWord = stoppedSentence.get(2);
			for (int i = 3; i < stoppedSentence.size(); i++) {
				String word = stoppedSentence.get(i);
				wordCounter.incrementCount(word, 1.0);
				bigramCounter.incrementCount(previousWord, word, 1.0);
				trigramCounter.incrementCount(prePreviousWord + " " + previousWord,
						word, 1.0);
				quadgramCounter.incrementCount(prePrePreviousWord + " " + prePreviousWord + " " + previousWord,
						word, 1.0);
				prePrePreviousWord = prePreviousWord;
				prePreviousWord = previousWord;
				previousWord = word;
			}
		}
		wordCounter.incrementCount(UNKNOWN, 1.0);
		normalizeDistributions();
	}

	private void normalizeDistributions() {
		for (String previousTrigram : quadgramCounter.keySet()) {
			quadgramCounter.getCounter(previousTrigram).normalize();
		}
		for (String previousBigram : trigramCounter.keySet()) {
			trigramCounter.getCounter(previousBigram).normalize();
		}
		for (String previousWord : bigramCounter.keySet()) {
			bigramCounter.getCounter(previousWord).normalize();
		}
		wordCounter.normalize();
	}
}
