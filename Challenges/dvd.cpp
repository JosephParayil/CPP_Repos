#include <iostream>
using namespace std;

void problem() {
    unsigned int px,py,sx,sy; cin >> px >> py >> sx >> sy;
    px--; py--;
    unsigned long long dist = 0;
    bool dX = true;
    bool dY = false;
    sx-=px;
    sy-=py;
    unsigned int x = 0;
    unsigned int y = 0;

    do
    {
        if (dX&&x==sx-1) dX = false;
        else if ((!dX)&&x==0) dX = true;
        if (dY&&y==sy-1) dY = false;
        else if ((!dY)&&y==0) dY = true;

        if (dX) x++;
        else x--;
        if (dY) y++;
        else y--;
        dist++;
    } while (!((x==0||x==sx-1)&&(y==0||y==sy-1)));
    
    cout << dist << '\n';
}

int main() {
    int t = 0; cin >> t;
    while (t--) problem();
    return 0;
}