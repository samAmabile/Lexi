//Parses pre-2022 code from GITHUB and STACK OVERFLOW for ML training
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <regex>
#include <mutex>
#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <atomic> 
#include <unordered_map>
#include <algorithm>
#include <cctype>
#include <iterator>
#include <filesystem> 
#include <cstdlib>


const char* git_ptr = std::getenv("GITHUB_TOKEN");
std::string git_token(git_ptr);

std::ofstream* csv_out = nullptr;

std::atomic<int> global_script_count(0); 

const int COLLECTION_LIMIT = 20000; 

using json = nlohmann::json;
namespace fs = std::filesystem;

std::mutex file_mutex;

const std::regex SYS_FILTER(
        R"(\.(json|md|txt|yaml|ini|confi|cfg|sh|bat|cmake|makefile|am|in|out|log|csv|css|html|zip|tar|bin|joblib|pickle)$)",
        std::regex_constants::icase
        );

const std::regex BINARY_NOISE_FILTER(
    R"(^(%?PDF|JFIF|\x89?PNG|\x7f?ELF|MZ|PK\x03\x04|GIF8[79]a|SQLite format 3|<\?xml|<!DOCTYPE html|<html))",
    std::regex_constants::icase
);

const std::regex MD_BLOCK(R"(```[a-zA-Z]*\n([\s\S]*?)\n```)"); 


std::string sanitize(const std::string& content){

    std::regex whitelist(R"([^a-zA-Z0-9""';:}{[\]()\*&#@!<>,./\\+=\^?`_ \t\n\r-])");
    std::string cleaned = std::regex_replace(content, whitelist, "");

    cleaned = std::regex_replace(cleaned, std::regex(R"(")"), R"("")");

    return cleaned; 
}

bool verify_py_regex(const std::string& code){
    const std::regex keyword_regex(R"(\b(def|for|while|import|elif|if|True|False|from|with|in|or|and)\b)");
    const std::regex comment(R"(#\w+)");
    double score = 0.0; 
    if (std::regex_search(code, comment)) score += 2.0;
    
    auto begin = std::sregex_iterator(code.begin(), code.end(), keyword_regex);
    auto end = std::sregex_iterator();

    score += std::distance(begin, end); 

    size_t lines = std::count(code.begin(), code.end(), '\n') +1;

    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.5; 
    
}

bool verify_cpp_regex(const std::string& code){
    const std::regex keyword_regex(R"(\b(int|float|double|char|void|std|cout|cin|include|return|const|struct|class|public|private|protected|namespace|using|auto|nullptr|true|false|for|while|do|switch)\b)");
    const std::regex comment(R"(//.*|/\*[\s\S]*?\*/)");
    double score = 0.0; 
    if (std::regex_search(code, comment)) score += 2.0;
    
    auto begin = std::sregex_iterator(code.begin(), code.end(), keyword_regex);
    auto end = std::sregex_iterator();

    score += std::distance(begin, end); 
    size_t semicolons = std::count(code.begin(), code.end(), ';'); 
    score += static_cast<double>(semicolons); 

    size_t lines = std::count(code.begin(), code.end(), '\n') +1;

    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65; 
    
}


bool verify_c_regex(const std::string& code){
    const std::regex keyword_regex(R"(\b(int|long|short|float|double|char|void|struct|union|enum|return|const|volatile|static|register|sizeof|typedef|if|else|for|while|do|switch|case|default|break|continue|malloc|free|stdio.h|stdlib.h)\b)");
    const std::regex comment(R"(//.*|/\*[\s\S]*?\*/)");
    double score = 0.0; 
    if (std::regex_search(code, comment)) score += 2.0;
    
    auto begin = std::sregex_iterator(code.begin(), code.end(), keyword_regex);
    auto end = std::sregex_iterator();

    score += std::distance(begin, end); 

    size_t semicolons = std::count(code.begin(), code.end(), ';'); 
    score += static_cast<double>(semicolons); 

    size_t lines = std::count(code.begin(), code.end(), '\n') +1;

    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65; 
    
}

bool verify_java_regex(const std::string& code){
    const std::regex keyword_regex(R"(\b(public|private|protected|class|interface|extends|implements|static|final|void|new|this|super|import|package|return|if|else|for|while|try|catch|finally|throw|throws|boolean|int|double|float|long|char|byte|short)\b)");
    const std::regex comment(R"(//.*|/\*[\s\S]*?\*/)");
    double score = 0.0; 
    if (std::regex_search(code, comment)) score += 2.0;
    
    auto begin = std::sregex_iterator(code.begin(), code.end(), keyword_regex);
    auto end = std::sregex_iterator();

    score += std::distance(begin, end); 

    size_t semicolons = std::count(code.begin(), code.end(), ';'); 
    score += static_cast<double>(semicolons); 

    size_t lines = std::count(code.begin(), code.end(), '\n') +1;

    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65; 
    
}

bool verify_py(const std::string& code){
    const std::vector<std::string> keywords = {
        "def ", "for ", "while ", "import ", "elif ", "if ", "True", "False", "from ", "import ", " in ", " and ", " or ", " with ", "return", "break",
        "try", "except", "Exception"
    };

    double score = 0.0; 

    if (code.find('#') != std::string::npos) score += 2; 

    for (const auto& word : keywords){
        size_t pos = code.find(word); 
        while (pos != std::string::npos){
            score += 1.0; 
            pos = code.find(word, pos + word.length());
        }
    }

    size_t lines = std::count(code.begin(), code.end(), '\n') + 1;
    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.5;

}

bool verify_cpp(const std::string& code){

    //(int|float|double|char|void|std|cout|cin|include|return|const|struct|class|public|private|protected|namespace|using|auto|nullptr|true|false|for|while|do|switch);

    const std::vector<std::string> keywords = {
        "int ", "float ", "double ", "char ", "void ", "std", "cout", "cin", "#include", "return", "const ", "struct ", "class ", "public", "private", "protected", "namespace", "using ", "auto ",
        "nullptr", "true", "false", "for", "while", "do", "switch", "size_t", "bool", "break", "continue"
    };

    double score = 0.0; 

    if (code.find("//") != std::string::npos || code.find("/*") != std::string::npos) score += 2; 
    size_t semicolons = std::count(code.begin(), code.end(), ';'); 

    score += static_cast<double>(semicolons);

    for (const auto& word : keywords){
        size_t pos = code.find(word); 
        while (pos != std::string::npos){
            score += 1.0; 
            pos = code.find(word, pos + word.length());
        }
    }

    size_t lines = std::count(code.begin(), code.end(), '\n') + 1;
    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65;

}

bool verify_c(const std::string& code){


    const std::vector<std::string> keywords = {
        "int ", "float ", "double ", "char ", "void ", "std", "cout", "cin", "#include", "return", "const ", "struct ", "namespace", "using ",
        "true", "false", "for", "while", "do", "switch", "size_t", "bool", "malloc", "free", "union", "enum", "volatile", "sizeof", "typedef", "stdio.h", "stdlib.h",
        "break"
    };

    double score = 0.0; 

    if (code.find("//") != std::string::npos || code.find("/*") != std::string::npos) score += 2; 
    size_t semicolons = std::count(code.begin(), code.end(), ';'); 

    score += static_cast<double>(semicolons);

    for (const auto& word : keywords){
        size_t pos = code.find(word); 
        while (pos != std::string::npos){
            score += 1.0; 
            pos = code.find(word, pos + word.length());
        }
    }

    size_t lines = std::count(code.begin(), code.end(), '\n') + 1;
    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65;

}

bool verify_java(const std::string& code){

    const std::vector<std::string> keywords = {
        "public", "private", "protected", "class", "interface", "extends", "implements", "static", "final", "void", "new", "this", "super", "import", "package",
        "return", "if", "else", "for", "while", "try", "catch", "finally", "throw ", "throws", "boolean", "int ", "double ", "float ", "long ", "char ", "byte", "short"
        };

    double score = 0.0; 

    if (code.find("//") != std::string::npos || code.find("/*") != std::string::npos) score += 2; 
    size_t semicolons = std::count(code.begin(), code.end(), ';'); 

    score += static_cast<double>(semicolons);

    for (const auto& word : keywords){
        size_t pos = code.find(word); 
        while (pos != std::string::npos){
            score += 1.0; 
            pos = code.find(word, pos + word.length());
        }
    }

    size_t lines = std::count(code.begin(), code.end(), '\n') + 1;
    double ratio = score / static_cast<double>(lines); 

    return ratio >= 0.65;

}



bool verify_code(const std::string& code, const std::string& lang){

    if (lang == "py"){
        return verify_py(code); 
    }
    if (lang == "cpp"){
        return verify_cpp(code);
    }
    if (lang == "c"){
        return verify_c(code);
    }
    if (lang == "java"){
        return verify_java(code);
    }

    return false;
}
    

void append_csv(const std::string& lang, const std::string& content){
    std::string esc = sanitize(content);



    if (esc.length() < 40) return;


    std::lock_guard<std::mutex> lock(file_mutex); 
    if (csv_out && csv_out->is_open()){
        (*csv_out) << "\"" << lang << "\",\"" << esc << "\"\n";
    }
}

void get_repo(const std::string& url, const std::string& lang){
    try{
        cpr::Response res = cpr::Get(cpr::Url{url + "/contents"},
                            cpr::Header{
                            {"User-Agent", "cpp-stylometric-crawler"},
                            {"Authorization", "token " + git_token},
                            {"Accept", "application/vnd.github+json"},
                            {"X-GitHub-Api-Version", "2022-11-28"}});
        if (res.status_code != 200) return;

        try{
            auto files = json::parse(res.text); 
            int num_files = 0; 

            for (const auto& file : files) { 
                if (global_script_count >= COLLECTION_LIMIT){
                    std::cout << "scraping goal reached: " << COLLECTION_LIMIT << std::endl;
                    return;
                }
                if (file.value("type", "") != "file") continue; 

                if (!file.contains("download_url") || file["download_url"].is_null()) {
                    continue; 
                }

                std::string path = file.value("path", ""); 
                if (std::regex_search(path, SYS_FILTER)) continue; 

                std::string download_url = file.value("download_url", ""); 
                cpr::Response raw_file = cpr::Get(cpr::Url{download_url}); 

                if (raw_file.status_code == 200 && !raw_file.text.empty()){
                    if (raw_file.text.length() > 100000) continue;

                    if (std::regex_search(
                            raw_file.text.begin(), 
                            raw_file.text.begin() + std::min(raw_file.text.length(), size_t(50)), 
                            BINARY_NOISE_FILTER)
                    ) continue;
                    
                    if (!verify_code(raw_file.text, lang)) continue;

                    append_csv(lang, raw_file.text);
                    int cur = ++global_script_count; 
                    if (cur >= COLLECTION_LIMIT){
                        std::cout << "scraping goal reached: " << COLLECTION_LIMIT <<std::endl;
                        return;
                    }
                    num_files++; 
                    if (num_files > 9) break; 

                }
            } 
        } catch (...) {}
    } catch (const std::exception& e){
        std::lock_guard<std::mutex> lock(file_mutex); 
        std::cerr << "Runtime error, skipping file..." << e.what() << std::endl;
    }

}

void get_stackex_code(const std::string& api_url, const std::string& lang){
    try{
        cpr::Response sr = cpr::Get(cpr::Url{api_url},cpr::Header{
                                        {"User-Agent", "cpp_stylometric_scraper"}, 
                                        {"Accept-Encoding", "gzip, deflate"}}
                );


        if (sr.status_code != 200) return; 

        try{
            auto json_data = json::parse(sr.text); 

            for (const auto& item : json_data["items"]){

                if (!item.contains("body") || item["body"].is_null()) {
                    continue;
                }
                std::string page_content = item.value("body", ""); 
                if (page_content.empty()) continue; 
                

                std::smatch match; 
                std::string::const_iterator search_start(page_content.cbegin()); 
                std::string::const_iterator search_end(page_content.cend());

                while (std::regex_search(search_start, search_end, match, MD_BLOCK)) {
                    std::string code_block = match[1].str(); 

                    if (code_block.length() > 60 && verify_code(code_block, lang)){
                        append_csv(lang, code_block);
                        global_script_count++; 
                    }

                    search_start = match.suffix().first; 
                }
            }
        } catch (...) {}
    } catch (const std::exception& e){
        std::lock_guard<std::mutex> lock(file_mutex); 
        std::cerr << "Runtime error, skipping file ... " << e.what() << std::endl;
    }
}

void lower(std::string& s){

    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { 
            return std::tolower(c); 
    });
}
            
int main(int argc, char* argv[]){
    
    fs::path exe = fs::absolute(fs::path(argv[0]));
    fs::path pwd = exe.parent_path(); 
    fs::path root = pwd.parent_path(); 
    fs::path csv_path = root / "data" / "_webscraped_code.csv";

    //TODO: finish resolving paths

    std::vector<std::string> query_keywords = {
        "homework+language:python", "algorithms+language:python", "dsa+language:python", "coursework+language:python",
        "assignment+language:cpp", "homework+language:cpp", "dsa+language:cpp", "coursework+language:cpp",
        "university+language:java", "dsa+language:java", "oop+language:java", "algorithms+language:java"
    };
    
    std::ofstream csv_file(csv_path, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "Could not resolve file path " << csv_path << std::endl;
        return 1; 
    }


    csv_out = &csv_file;
    
    if (fs::file_size(csv_path) == 0){
        (*csv_out) << "lang,src\n"; 
    }

    std::vector<std::string> github_queries = {
            "https://api.github.com/search/repositories?q=homework+language:python+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=algorithms+language:python+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=dsa+language:python+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=coursework+language:python+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=assignment+language:cpp+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=homework+language:cpp+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=dsa+language:cpp+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=coursework+language:cpp+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=university+language:java+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=dsa+language:java+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=oop+language:java+pushed:<2022-01-01&per_page=100&sort=stars",
            "https://api.github.com/search/repositories?q=algorithms+language:java+pushed:<2022-01-01&per_page=100&sort=stars",
    };
    const char* stack_ptr = std::getenv("STACK_KEY");
    std::string stack_key(stack_ptr);
    std::vector<std::string> stackex_urls = {
            "https://api.stackexchange.com/2.3/answers?todate=1640995200&order=desc&sort=votes&tagged=cpp&site=stackoverflow&filter=!nNPvRIY_Y6&key="+stack_key,
            "https://api.stackexchange.com/2.3/answers?todate=1640995200&order=desc&sort=votes&tagged=c&site=stackoverflow&filter=!nNPvRIY_Y6&key="+stack_key,
            "https://api.stackexchange.com/2.3/answers?todate=1640995200&order=desc&sort=votes&tagged=python&site=stackoverflow&filter=!nNPvRIY_Y6&key="+stack_key,
            "https://api.stackexchange.com/2.3/answers?todate=1640995200&order=desc&sort=votes&tagged=java&site=stackoverflow&filter=!nNPvRIY_Y6&key="+stack_key
    };
    
    std::vector<std::string> accepted_langs = { "python", "py", "c++", "cpp", "c-plus-plus", "c", "java" };
    std::unordered_map<std::string, std::string> langmap = {
        {"python", "py"}, 
        {"py", "py"}, 
        {"c++", "cpp"},
        {"cpp", "cpp"},
        {"c-plus-plus", "cpp"},
        {"c", "c"}, 
        {"java", "java"}
    };


    std::vector<std::future<void>> callers; 

    std::vector<std::string> stack_langs = {"cpp", "c", "py", "java"};

    for (size_t i = 0; i < stackex_urls.size(); i++){
        callers.push_back(std::async(std::launch::async, get_stackex_code, stackex_urls[i], stack_langs[i]));
    }

    for (const auto& keyword : query_keywords) {
        for (int i = 1; i <= 5; i++){

            if (global_script_count.load() >= COLLECTION_LIMIT) break;

            std::string api_url = "https://api.github.com/search/repositories?q=" + keyword + 
                                  "+pushed:<2022-01-01&per_page=100&sort=stars&page=" + std::to_string(i);

            cpr::Response qr = cpr::Get(cpr::Url{api_url}, cpr::Header{
                                                                {"User-Agent", "cpp-stylometric-crawler"}, 
                                                                {"Authorization", "token " + git_token}});

            if (qr.status_code != 200) continue; 
            
            try{
                auto data = json::parse(qr.text);

                if (!data.contains("items") || !data["items"].is_array() || data["items"].is_null()){
                    std::cerr << "Unexpected return from github, may be rate limiting, invalid key, blocking bot access" << std::endl;
                    continue;
                }
                for (const auto& repo : data["items"]){
                    
                    std::string url = repo.value("url", "");

                    if (!repo.contains("language") || repo["language"].is_null()) continue;
                    std::string raw_lang = repo.value("language", "unknown"); 
                    lower(raw_lang);
                    
                    bool is_accepted = false; 
                    for (const std::string& s : accepted_langs){
                        if (raw_lang == s){
                            is_accepted = true; 
                            break;
                        }
                    }

                    if (!is_accepted) continue;
                    
                    std::string lang = langmap[raw_lang];

                    callers.push_back(std::async(std::launch::async, get_repo, url, lang)); 
                }
            } catch(...) {
                std::cerr << "Failed to parse JSON root" << std::endl;
            }
            for (auto& task : callers) {
                if (task.valid()) task.wait();
            }
            callers.clear();
        }

    }
    
    for (auto& task : callers) {
        task.wait(); 
    }

    std::cout << "data collection round complete, csv updated safely." <<std::endl;
    return 0; 

}
                                            

