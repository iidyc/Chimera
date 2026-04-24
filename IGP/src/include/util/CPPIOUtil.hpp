//
// Created by Administrator on 2025/5/19.
//

#ifndef CPPIOUTIL_HPP
#define CPPIOUTIL_HPP
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

namespace VectorSetSearch {
inline std::string getDataRoot() {
  const char* env = std::getenv("IGP_LOCAL_DATA_ROOT");
  if (env != nullptr && env[0] != '\0') {
    return std::string(env);
  }
  return ".";
}

inline std::vector<uint32_t> readQueryID(const std::string /*username*/,
                                         const std::string dataset,
                                         const uint32_t n_query) {
  std::stringstream ss;
  ss << getDataRoot() << "/RawData/" << dataset << "/document/queries.dev.tsv";
  const std::string location = ss.str();

  std::vector<uint32_t> qID_l;
  std::ifstream file(location);

  if (!file.is_open()) {
    qID_l.reserve(n_query);
    for (uint32_t qid = 0; qid < n_query; ++qid) {
      qID_l.push_back(qid);
    }
    return qID_l;
  }

  std::string line;
  while (getline(file, line)) {
    if (line.empty()) continue;

    size_t tab_pos = line.find('\t');
    if (tab_pos == std::string::npos) {
      std::cerr << "Warning: Invalid format in line: " << line << std::endl;
      continue;
    }

    std::string id = line.substr(0, tab_pos);
    qID_l.push_back(stoi(id));
  }

  return qID_l;
}
}
#endif //CPPIOUTIL_HPP
