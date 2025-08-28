#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <random>

/// @brief Configuration structure for CDPruner algorithm
struct Config
{
	/// @brief Percentage of visual tokens to retain after pruning (0-100)
	size_t visual_tokens_retain_percentage = 60;

	/// @brief Weight for balancing relevance vs diversity (0.0 to 1.0)
	float relevance_weight = 0.5f;

	/// @brief Whether to enable pruning functionality
	bool enable_pruning = true;

	/// @brief Device to run CDPruner computations on
	std::string device = "CPU";

	/// @brief Whether to enable debug output
	bool pruning_debug_mode = false;

	/// @brief Threshold for numerical stability
	float numerical_threshold = 1e-6f;

	/// @brief Whether to apply negative mean for relevance calculation
	/// This is needed for CLIP-based models (like LLaVA) due to counterintuitive similarity values
	bool use_negative_relevance = false;

	/// @brief Whether to use OpenVINO ops model for computation
	/// When true, uses integrated OpenVINO ops model for relevance and kernel computation
	/// When false, uses traditional step-by-step computation pipeline
	bool use_ops_model = false;

	/// @brief Compare two Config structures for equality
	/// @param other The other Config to compare with
	/// @return true if all configuration parameters are equal, false otherwise
	bool operator==(const Config &other) const
	{
		return visual_tokens_retain_percentage == other.visual_tokens_retain_percentage &&
			   std::abs(relevance_weight - other.relevance_weight) < 1e-6f && enable_pruning == other.enable_pruning &&
			   device == other.device && pruning_debug_mode == other.pruning_debug_mode &&
			   std::abs(numerical_threshold - other.numerical_threshold) < 1e-9f &&
			   use_negative_relevance == other.use_negative_relevance;
	}

	/// @brief Compare two Config structures for inequality
	/// @param other The other Config to compare with
	/// @return true if any configuration parameters differ, false otherwise
	bool operator!=(const Config &other) const
	{
		return !(*this == other);
	}
};

struct Tensor {
	float* data = nullptr;
	int b = 0;
	int m = 0;
	int n = 0;
	Tensor(int s1, int s2, int s3)
	{
		b = s1;
		m = s2;
		n = s3;
		data = (float *)malloc(b * m * n * sizeof(float));
		if (nullptr == data)
		{
			std::cout << "Error: cant't malloc size[" << b * m * n << "]." << std::endl;
			return;
		}
	}

	Tensor(int s2, int s3) : Tensor(1, s2, s3) {}
	Tensor(int s3) : Tensor(1, 1, s3) {}
	~Tensor()
	{
		if (data)
		{
			free(data);
			data = nullptr;
			b = m = n = 0;
		}
	}

	size_t get_byte_size() {
		return b * m * n * sizeof(float);
	}
	size_t get_size() const
	{
		return b * m * n;
	}

	void random_data() {
		std::random_device rd; 
		std::mt19937 gen(rd());
		std::uniform_real_distribution<float> dis(0.0f, 1.0f);
		for (int i = 0; i < b * m * n; i++)
		{
			data[i] = dis(gen);
		}
	}
};

class FastGreedyDPP
{
public:
	/// @brief Constructor
	/// @param config Configuration for the DPP selector
	explicit FastGreedyDPP(const Config &config);

	/**
	 * @brief Select diverse tokens using fast greedy DPP algorithm
	 * @param kernel Conditional kernel matrix [B, N, N]
	 * @param num_tokens Number of tokens to select
	 * @return Selected token indices for each batch [B, T]
	 */
	std::vector<std::vector<size_t>> select(const Tensor &kernel, size_t num_tokens);

	/**
	 * @brief Create boolean mask from selected indices
	 * @param selected_indices Selected indices for each batch [B, T]
	 * @param total_tokens Total number of tokens
	 * @return Boolean mask [B*N] where true indicates selected tokens
	 */
	static std::vector<bool> create_mask(const std::vector<std::vector<size_t>> &selected_indices,
										 size_t total_tokens);

	/**
	 * @brief Compute approximate determinant for validation
	 * @param kernel Kernel matrix [1, N, N] (single batch only)
	 * @param selected_indices Selected token indices
	 * @return Approximated determinant value
	 */
	static float compute_determinant_approximation(const Tensor &kernel,
												   const std::vector<size_t> &selected_indices);

private:
	/**
	 * @brief Select tokens for a single batch
	 * @param kernel Kernel matrix [B, N, N]
	 * @param batch_idx Batch index to process
	 * @param num_tokens Number of tokens to select
	 * @return Selected token indices for this batch
	 */
	std::vector<size_t> select_single_batch(const Tensor &kernel, size_t batch_idx, size_t num_tokens);

	/**
	 * @brief Find index with maximum value
	 * @param scores Score tensor [N]
	 * @return Index of maximum value
	 */
	size_t argmax(const Tensor &scores);

	/**
	 * @brief Update orthogonal vector using Gram-Schmidt process
	 * @param kernel Kernel matrix [B, N, N]
	 * @param batch_idx Current batch index
	 * @param selected_idx Newly selected token index
	 * @param iteration Current iteration (number of previously selected tokens)
	 * @param cis Orthogonalized vectors [T, N]
	 * @param di2s Current diagonal scores [N]
	 */
	void update_orthogonal_vector(const Tensor &kernel, size_t batch_idx, size_t selected_idx,
								  size_t iteration, Tensor &cis, const Tensor &di2s);

	/**
	 * @brief Update marginal gains after selecting a token
	 * @param iteration Current iteration
	 * @param selected_idx Newly selected token index
	 * @param cis Orthogonalized vectors [T, N]
	 * @param di2s Diagonal scores to update [N]
	 */
	void update_marginal_gains(size_t iteration, size_t selected_idx,
							   const Tensor &cis, Tensor &di2s);

	Config m_config;
};