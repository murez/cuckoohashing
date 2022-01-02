#ifndef __CSVWRITER_H__
#define __CSVWRITER_H__

#include <fstream>
#include <sstream>
#include <vector>

namespace csv {

class Record {
 private:
  std::vector<std::string> items;
  char seperator = ',';

 public:
  Record() = default;
  Record(char seperator) : seperator(seperator) {}
  Record(const std::vector<std::string>& record) : items(record) {}
  Record(const std::vector<std::string>& record, char seperator)
      : items(record), seperator(seperator) {}
  ~Record() = default;

  template <typename T>
  void put(const T& val) {
    std::stringstream ss;
    ss << val;
    items.push_back(ss.str());
  }

  std::string toString() {
    std::stringstream ss;
    for (auto& item : items) {
      if (&item == &*items.rbegin()) {
        ss << item;
      } else {
        ss << item << seperator;
      }
    }
    return ss.str();
  }

  bool empty() { return items.empty(); }
};

class CsvWriter {
 private:
  std::fstream file;
  std::string filePath;

  Record header;
  std::vector<Record> records;

 public:
  CsvWriter(const std::string& filePath) : filePath(filePath) {
    file.open(filePath, std::ios::out);
  }

  ~CsvWriter() { close(); }

  void setHeader(const Record& header) { this->header = header; }

  void insertRecord(const Record& record) { records.push_back(record); }

  bool isOpen() { return file.is_open(); }

  void write() {
    if (!header.empty()) {
      file << header.toString() << std::endl;
    }

    for (auto& record : records) {
      file << record.toString() << std::endl;
    }
  }

  void write(Record& record) {
    if (isOpen()) {
      file.open(filePath, std::ios::out | std::ios::app);
    }
    file << record.toString() << std::endl;
  }

  void close() {
    if (file.is_open()) {
      file.close();
    }
  }
};

} /* csv */

#endif  // __CSVWRITER_H__