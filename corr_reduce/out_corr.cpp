#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
//#include "prettyprint.hpp"

using namespace std;

int main(int argc, char* argv[]){
  double threshold=0.99;
  if(argv[2]) threshold=stod(argv[2]);


  ofstream ofl(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmcorr.txt");
  ifstream ifs {argv[1]};
  vector<vector<string>> table; ///> the csv file values
  for(string line; getline(ifs,line);){
    stringstream ssline(line);
    vector<string> row;
    for(string cell; getline(ssline,cell,',');){
      row.push_back(cell);
    }
    table.push_back(row);
  }
  //ofl << table.size() << ", " << table.at(0).size() << endl;


  vector<size_t> exclude_columns; ///> columns that have correlation more then a threashhold but are not the diagonal elements
  vector<size_t> include_columns={0};
  for(size_t row=0; row<table.size();++row){
    vector<string> matched;
    matched.push_back(table[row].front());
    vector<int> correlated;
    for(size_t i=row; i<table[row].size();++i){
      if(find(begin(exclude_columns),end(exclude_columns),i)==exclude_columns.end()){
        if(abs(strtod(table[row][i].c_str(),nullptr))>threshold){
          matched.push_back(table[row][i]);
          if(i>row)
            exclude_columns.push_back(i);
          else
            include_columns.push_back(i);
          correlated.push_back(i);
        }
      }
    }
    if(correlated.size()>1){
      ofl << correlated.size()-1 << " ";
      for(auto i:correlated) ofl << i << " " << table.front()[i]<<" ";
      //ofl << endl;
      ofl << " -- ";
    }
    if(matched.size()>2) {
      for(auto i:matched) ofl << i << " ";
      ofl << endl;
    }
  }
  ofl << "total " << exclude_columns.size()<< endl;
  for(auto i : exclude_columns) ofl << i << " ";
  ofl << endl;
  //ofl << exclude_columns<< endl;


  ofstream ofindin(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmindexin.txt");
  for(auto i : include_columns) ofindin << i << endl;
  ofstream ofnamin(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmnamesin.txt");
  for(auto i : include_columns) ofnamin << table.front()[i] << endl;
  ofstream ofindex(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmindexex.txt");
  for(auto i : exclude_columns) ofindex << i << endl;
  ofstream ofnames(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmnamesex.txt");
  for(auto i : exclude_columns) ofnames << table.front()[i] << endl;
  ofstream ofs(string(argv[0])+string(argv[1])+to_string(threshold)+"_rmcorr.csv");
  //ofstream ofs(string(argv[0])+string(argv[1])+string(argv[2])+"_rmcorr.csv");
  cout << table.size() << ", " << table.at(0).size() << endl;
  for(size_t row=0; row<table.size();++row){
    if(find(begin(exclude_columns),end(exclude_columns),row)==exclude_columns.end()){
      for(size_t i=0; i<table[row].size();++i){
        if(find(begin(exclude_columns),end(exclude_columns),i)==exclude_columns.end()){
          ofs << table[row][i] << ',';
        }
      }
      ofs << endl;
    }
  }
  return 0;
}
