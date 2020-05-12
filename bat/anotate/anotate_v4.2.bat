@echo off
rem ファイル名の一覧を取得

rem ==================================
echo "+++++ 確認事項 +++++"
echo "+"
echo "+ OK, NG, AnnotationMissのフォルダはありますか？"
echo "+ 画像はwindowsで設定されているアプリが起動します"
echo "+ (エラーが発生する場合は管理者で実行してみてください)"
echo "+"
echo "+++++++++++++++"

rem ==================================
rem ラベルとキーボードとの割り当て
set LABEL_OK=1
set LABEL_NG=3
set LABEL_AnnotationMiss=5

rem ==================================
echo "処理するファイル数:"
for  %%A in (*.JPG) do ( if exist %%A (set /a counter=counter+1) )
echo %counter%
set i=0

rem ==================================
rem Mainの処理開始
SETLOCAL enabledelayedexpansion
for %%A in (*.JPG) do (
	
	echo "=============================="
	echo "現在の進捗:%i% / %counter%"
	rem ==================================
	rem 画像の描画
	echo "ファイル名:"
        echo %%A
	%%A

	rem ==================================
	REM ユーザーからの入力を受け付ける
	set /p INPUT_LABEL="ラベルを入力してください \n(%LABEL_OK%:OK, %LABEL_NG%:NG, %LABEL_AnnotationMiss%:AnnotationMiss) >>>"
	echo INPUT_LABEL is !INPUT_LABEL! .

	rem ==================================
	REM 画像の移動
	if !INPUT_LABEL! == %LABEL_OK% (
		echo INPUT_LABEL is %LABEL_OK% .
		move %%A .\OK
	)

	if !INPUT_LABEL! == %LABEL_NG% (
		echo INPUT_LABEL is %LABEL_NG% .
		move %%A .\NG
	)

	if !INPUT_LABEL! == %LABEL_AnnotationMiss% (
		echo INPUT_LABEL is %LABEL_AnnotationMiss% .
		move %%A .\NG
	)

	rem ==================================
	REM 画像の描画終了
	taskkill /F /IM Microsoft.Photos.exe
)
ENDLOCAL
echo "+++++++++++++++"
echo "+"
echo "+ 終了です！！
echo "+ お疲れ様でした(о´∀`о)．
echo "+"
echo "+++++++++++++++"
pause