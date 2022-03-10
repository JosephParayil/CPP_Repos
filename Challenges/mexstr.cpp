#include <iostream>
using namespace std;


void problem() {
    string s; cin>>s;
    string mexstr = "_";
    for (char x='a'-1;x<='z';x++) {
        for (char y='a';y<='z';y++) {
            mexstr[mexstr.size()-1] = y;
            if (s.find(mexstr)==string::npos) { cout << mexstr << '\n'; return;}
        }
        if (x<'a') {
            mexstr[0]='a';
            mexstr.push_back('a');
        } else mexstr[0] = x;
    }
    
}

int main() {
    int t = 0; cin >> t;
    while (t--) problem();
    return 0;
}