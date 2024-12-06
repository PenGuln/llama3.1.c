#define TESTING
#include "run.c"

double chat_generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(32768 * sizeof(int));

    prompt_tokens[num_prompt_tokens++] = 128000; // "<|begin_of_text|>"
    prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
    prompt_tokens[num_prompt_tokens++] = 882;    // "user"
    prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
    prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"
    encode(tokenizer, prompt, 0, 0, prompt_tokens + num_prompt_tokens, &num_prompt_tokens);
    prompt_tokens[num_prompt_tokens++] = 128009; // "<|eot_id|>"
    prompt_tokens[num_prompt_tokens++] = 128006; // "<|start_header_id|>"
    prompt_tokens[num_prompt_tokens++] = 78191;  // "assistant"
    prompt_tokens[num_prompt_tokens++] = 128007; // "<|end_header_id|>"
    prompt_tokens[num_prompt_tokens++] = 271;    // "\n\n"

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if ((next == 128001 || next == 128009) && pos > num_prompt_tokens)
            break;

        // print the token as string, decode it with the Tokenizer object
        char* piece = decode(tokenizer, token, next);
        safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");
    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    double speed = 0;
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
        speed = (pos-1) / (double)(end-start)*1000;
    }
    free(prompt_tokens);
    return speed;
}

void error_usage() {
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 4096;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);
    
    char buffer[10000];
    FILE *ptr;
    ptr = fopen("data.bin", "rb");  // r for read, b for binary
    fread(buffer, sizeof(buffer), 1, ptr); // read 10 bytes to our buffer
    fclose(ptr);
    int i, idx = 0;
    double speed = 0;
    int n_outputs = 0;
    for (i = 0; i < 10000; i++) {
        if (buffer[i] == 0) {
            if (i > idx) {
                prompt = buffer + idx;
                speed += chat_generate(&transformer, &tokenizer, &sampler, prompt, steps);
                n_outputs += 1;
                idx = i + 1;
            }else{
                break;
            }
        }
    }
    fprintf(stderr, "average tok/s: %f\n", speed / n_outputs);
    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}